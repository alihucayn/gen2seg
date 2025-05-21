# gen2seg official inference pipeline code for Stable Diffusion model
# 
# This code was adapted from Marigold and Diffusion E2E Finetuning. 
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper, along with the two works above. 

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import PipelineImageInput
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.schedulers import (
    DDIMScheduler,
)
from diffusers.utils import (
    BaseOutput,
    logging,
)
from diffusers import DiffusionPipeline
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor

# add
def zeros_tensor(
    shape: Union[Tuple, List],
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """
    A helper function to create tensors of zeros on the desired `device`.
    Mirrors randn_tensor from diffusers.utils.torch_utils.
    """
    layout = layout or torch.strided
    device = device or torch.device("cpu")
    latents = torch.zeros(list(shape), dtype=dtype, layout=layout).to(device)
    return latents


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class gen2segSDSegOutput(BaseOutput):
    """
    Output class for gen2seg Instance Segmentation prediction pipeline.

    Args:
        prediction (`np.ndarray`, `torch.Tensor`):
            Predicted instance segmentation with values in the range [0, 255]. The shape is always $numimages \times 1 \times height
            \times width$, regardless of whether the images were passed as a 4D array or a list.
        latent (`None`, `torch.Tensor`):
            Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline.
            The shape is $numimages * numensemble \times 4 \times latentheight \times latentwidth$.
    """

    prediction: Union[np.ndarray, torch.Tensor]
    latent: Union[None, torch.Tensor]


class gen2segSDPipeline(DiffusionPipeline):
    """
    # add
    Pipeline for Instance Segmentation prediction using our Stable Diffusion model.  
    Implementation is built upon Marigold: https://marigoldmonodepth.github.io and E2E FThttps://gonzalomartingarcia.github.io/diffusion-e2e-ft/

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the segmentation latent, synthesized from image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent
            representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latent.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """
    
    model_cpu_offload_seq = "text_encoder->unet->vae"

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_processing_resolution: Optional[int] = 768, # add
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_processing_resolution=default_processing_resolution,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_processing_resolution = default_processing_resolution
        self.empty_text_embedding = None

        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def check_inputs(
        self,
        image: PipelineImageInput,
        processing_resolution: int,
        resample_method_input: str,
        resample_method_output: str,
        batch_size: int,
        output_type: str,
    ) -> int:
        if processing_resolution is None:
            raise ValueError(
                "`processing_resolution` is not specified and could not be resolved from the model config."
            )
        if processing_resolution < 0:
            raise ValueError(
                "`processing_resolution` must be non-negative: 0 for native resolution, or any positive value for "
                "downsampled processing."
            )
        if processing_resolution % self.vae_scale_factor != 0:
            raise ValueError(f"`processing_resolution` must be a multiple of {self.vae_scale_factor}.")
        if resample_method_input not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_input` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if resample_method_output not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_output` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        if output_type not in ["pt", "np"]:
            raise ValueError("`output_type` must be one of `pt` or `np`.")

        # image checks
        num_images = 0
        W, H = None, None
        if not isinstance(image, list):
            image = [image]
        for i, img in enumerate(image):
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                if img.ndim not in (2, 3, 4):
                    raise ValueError(f"`image[{i}]` has unsupported dimensions or shape: {img.shape}.")
                H_i, W_i = img.shape[-2:]
                N_i = 1
                if img.ndim == 4:
                    N_i = img.shape[0]
            elif isinstance(img, Image.Image):
                W_i, H_i = img.size
                N_i = 1
            else:
                raise ValueError(f"Unsupported `image[{i}]` type: {type(img)}.")
            if W is None:
                W, H = W_i, H_i
            elif (W, H) != (W_i, H_i):
                raise ValueError(
                    f"Input `image[{i}]` has incompatible dimensions {(W_i, H_i)} with the previous images {(W, H)}"
                )
            num_images += N_i

        return num_images

    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(**self._progress_bar_config)
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = False,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        output_type: str = "np",
        output_latent: bool = False,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`),
                `List[torch.Tensor]`: An input image or images used as an input for the instance segmentation task. For
                arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible
                by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or
                three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the
                same width and height.
            processing_resolution (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters passing a tensor of images.
            output_type (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction`. The accepted ßvalues are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`gen2segSDSegOutput`] instead of a plain tuple.

        # add
        E2E FT models are deterministic single step models involving no ensembling, i.e. E=1.
        """

        # 0. Resolving variables.
        device = self._execution_device
        dtype = self.dtype

        # Model-specific optimal default values leading to fast and reasonable results.
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution
        
        #print(image[0].size)
        #processing_resolution = 8 * round(max(image[0].size) / 8)

        # 1. Check inputs.
        num_images = self.check_inputs(
            image,
            processing_resolution,
            resample_method_input,
            resample_method_output,
            batch_size,
            output_type,
        )

        # 2. Prepare empty text conditioning.
        # Model invocation: self.tokenizer, self.text_encoder.
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]

        # 3. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
        # optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
        # `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
        # divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
        # of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
        # operation and leads to the most reasonable results. Using the native image resolution or any other processing
        # resolution can lead to loss of either fine details or global context in the output predictions.
        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution, resample_method_input, device, dtype
        )  # [N,3,PPH,PPW]
    #     image =(image+torch.abs(image.min()))
    #     image = image/(torch.abs(image.max())+torch.abs(image.min()))
    #    # prediction = prediction**0.5
    #     #prediction = torch.clip(prediction, min=-1, max=1)+1
    #     image = (image) * 2
    #     image = image - 1  
        # 4. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
        # ensemble members. Each ensemble member is an independent diffused prediction, just initialized independently.
        # Latents of each such predictions across all input images and all ensemble members are represented in the
        # `pred_latent` variable. The variable `image_latent` is of the same shape: it contains each input image encoded
        # into latent space and replicated `E` times. Encoding into latent space happens in batches of size `batch_size`.
        # Model invocation: self.vae.encoder.
        image_latent, pred_latent = self.prepare_latents(
            image, batch_size
        )  # [N*E,4,h,w], [N*E,4,h,w]

        del image

        batch_empty_text_embedding = self.empty_text_embedding.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )  # [B,1024,2]

        # 5. Process the denoising loop. All `N * E` latents are processed sequentially in batches of size `batch_size`.
        # The unet model takes concatenated latent spaces of the input image and the predicted modality as an input, and
        # outputs noise for the predicted modality's latent space.
        # Model invocation: self.unet.
        pred_latents = []

        for i in range(0, num_images, batch_size):
            batch_image_latent = image_latent[i : i + batch_size]  # [B,4,h,w]
            batch_pred_latent = batch_image_latent[i : i + batch_size]    # [B,4,h,w]
            effective_batch_size = batch_image_latent.shape[0]
            text = batch_empty_text_embedding[:effective_batch_size]  # [B,2,1024]
            
            # add
            # Single step inference for E2E FT models
            self.scheduler.set_timesteps(1, device=device)
            for t in self.scheduler.timesteps:
                batch_latent = batch_image_latent # torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
                noise = self.unet(batch_latent, t, encoder_hidden_states=text, return_dict=False)[0]  # [B,4,h,w]
                batch_pred_latent = self.scheduler.step(
                    noise, t, batch_image_latent
                ).pred_original_sample  # [B,4,h,w], # add
                                                     # directly take pred_original_sample rather than prev_sample

            pred_latents.append(batch_pred_latent)

        pred_latent = torch.cat(pred_latents, dim=0)  # [N*E,4,h,w]

        del (
            pred_latents,
            image_latent,
            batch_empty_text_embedding,
            batch_image_latent,
          #  batch_pred_latent,
            text,
            batch_latent,
            noise,
        )

        # 6. Decode predictions from latent into pixel space. The resulting `N * E` predictions have shape `(PPH, PPW)`,
        # which requires slight postprocessing. Decoding into pixel space happens in batches of size `batch_size`.
        # Model invocation: self.vae.decoder.
        prediction = torch.cat(
            [
                self.decode_prediction(pred_latent[i : i + batch_size])
                for i in range(0, pred_latent.shape[0], batch_size)
            ],
            dim=0,
        )  # [N*E,1,PPH,PPW]

        if not output_latent:
            pred_latent = None

        # 7. Remove padding. The output shape is (PH, PW).
        prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,1,PH,PW]

        # 9. If `match_input_resolution` is set, the output prediction are upsampled to match the
        # input resolution `(H, W)`. This step may introduce upsampling artifacts, and therefore can be disabled.
        # Depending on the downstream use-case, upsampling can be also chosen based on the tolerated artifacts by
        # setting the `resample_method_output` parameter (e.g., to `"nearest"`).
        if match_input_resolution:
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [N,1,H,W]

        # 10. Prepare the final outputs.
        if output_type == "np":
            prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,1]

        # 11. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (prediction, pred_latent)

        return gen2segSDSegOutput(
            prediction=prediction,
            latent=pred_latent,
        )

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def retrieve_latents(encoder_output):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        image_latent = torch.cat(
            [
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # [N,4,h,w]
        image_latent = image_latent * self.vae.config.scaling_factor # [N*E,4,h,w]

        # add
        # provide zeros as noised latent
        pred_latent = zeros_tensor(
            image_latent.shape,
            device=image_latent.device,
            dtype=image_latent.dtype,
        )  # [N*E,4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]
        #print(prediction.max())
        #print(prediction.min())

        prediction =(prediction+torch.abs(prediction.min()))
        prediction = prediction/(torch.abs(prediction.max())+torch.abs(prediction.min()))
        #prediction = prediction**0.5
        #prediction = torch.clip(prediction, min=-1, max=1)+1
        prediction = (prediction) * 255.0
        #print(prediction.max())
        #print(prediction.min())
        return prediction  # [B,1,H,W]