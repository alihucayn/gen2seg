#!/usr/bin/env python3
"""
Training script for document forgery detection using Stable Diffusion.
Modified from the original train.py to use RTMDataset or DocTamperDataset for document forgery detection.
"""

import os
import math
import argparse
import logging
from pickletools import uint8
import random
import shutil
import warnings
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# Import custom modules
from dataloaders.load import RTMDataset, DocTamperDataset
from util.loss import BinarySegmentationLoss
from util.unet_prep import replace_unet_conv_in

# Import custom learning rate scheduler
from util.lr_scheduler import IterExponential

# Import IoU metric for validation
from eval.metrics.iou_metric import DocumentForgeryEvaluator

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("diffusers >= 0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Training code for document forgery detection using Stable Diffusion.")

    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_total_iter_length", type=int, default=20000)

    # Stable diffusion training settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/model-finetuned",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=500, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=2, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=15, required=True, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means main process.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "TensorBoard log directory. Will default to output_dir/runs/CURRENT_DATETIME_HOSTNAME."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "The integration to report the results and logs to. Supported platforms are `tensorboard`"
            " (default) and `wandb` (log with `wandb`)."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using the `--resume_from_checkpoint` argument.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' --checkpointing_steps, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xFormers."
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default=None,
        help="Type of noise to add. If None, no noise is added.",
    )
    parser.add_argument(
        "--random_state_file",
        type=str,
        default=None,
        help="Path to random state file for reproducible training.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="gen2segdfs",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for more information see "
            "https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # Dataset selection
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="rtm",
        choices=["rtm", "doctamper"],
        help="Type of dataset to use: 'rtm' for RTMDataset or 'doctamper' for DocTamperDataset",
    )

    # RTMDataset specific arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing images and labels folders (for RTMDataset).",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="images",
        help="Directory containing training images (for RTMDataset).",
    )
    parser.add_argument(
        "--label_dir", 
        type=str,
        default="labels",
        help="Directory containing training labels (for RTMDataset).",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="train.txt",
        help="File containing list of training samples (for RTMDataset).",
    )

    # DocTamperDataset specific arguments
    parser.add_argument(
        "--lmdb_paths",
        type=str,
        nargs='+',
        default=None,
        help="Paths to LMDB files (for DocTamperDataset).",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset mode (for DocTamperDataset).",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Random seed for dataset (for DocTamperDataset).",
    )

    # Validation settings
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--val_data_root",
        type=str,
        default=None,
        help="Root directory for validation data (for RTMDataset).",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="images",
        help="Directory containing validation images (for RTMDataset).",
    )
    parser.add_argument(
        "--val_label_dir", 
        type=str,
        default="labels",
        help="Directory containing validation labels (for RTMDataset).",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default="val.txt",
        help="File containing list of validation samples (for RTMDataset).",
    )
    parser.add_argument(
        "--val_lmdb_paths",
        type=str,
        nargs='+',
        default=None,
        help="Paths to validation LMDB files (for DocTamperDataset).",
    )
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=0.5,
        help="Threshold for IoU evaluation.",
    )
    
    parser.add_argument("--validation_count", type=int, default=2000, help="Number of validation steps to run.")
    
    parser.add_argument("--multiple_compressions", action="store_true", help="Use multiple compressions for RTMDataset.")
    
    parser.add_argument("--wandb_image_interval", type=int, default=20, help="Interval for logging images to wandb.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def encode_image(vae, image, dtype=torch.float32):
    """Apply VAE Encoder to image with proper dtype conversion."""
    image = image.to(dtype=dtype)
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent

def decode_image(vae, latent, dtype=torch.float32):
    """Apply VAE Decoder to latent with proper dtype conversion."""
    latent = latent.to(dtype=dtype)
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image

def evaluate_validation(model, vae, val_dataloader, accelerator, weight_dtype, args):
    """
    Evaluate model on validation set using IoU metrics.
    
    Args:
        model: UNet model
        vae: VAE model
        val_dataloader: Validation dataloader
        accelerator: Accelerator instance
        weight_dtype: Weight dtype
        args: Training arguments
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    evaluator = DocumentForgeryEvaluator(threshold=args.eval_threshold)
    
    # Pre-compute empty text CLIP encoding
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.to(accelerator.device)
    
    
    empty_token = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)
    
    # For converting from predicted noise -> latents
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_prod = 1 - alpha_prod
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        val_count = -1
        total = len(val_dataloader) if val_dataloader is not None and args.validation_count == -1 else args.validation_count
        for batch in tqdm(val_dataloader, desc="Validating", total=total, disable=not accelerator.is_local_main_process):
            val_count += 1
            if val_count >= total:  # Maximum number of validation iterations
                break
            # Encode RGB to latents
            rgb_latents = encode_image(
                vae,
                batch["image"].to(device=accelerator.device), dtype=weight_dtype
            )
            rgb_latents = rgb_latents * vae.config.scaling_factor

            # Timesteps
            timesteps = torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) \
                        * (noise_scheduler.config.num_train_timesteps - 1)
            timesteps = timesteps.long()
            noisy_latents = torch.zeros_like(rgb_latents)

            # UNet input
            encoder_hidden_states = empty_encoding.repeat(len(batch["image"]), 1, 1)
            unet_input = rgb_latents

            model_pred = model(
                unet_input,
                timesteps,
                encoder_hidden_states,
                return_dict=False
            )[0]

            alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
            beta_prod_t = beta_prod[timesteps].view(-1, 1, 1, 1)

            if noise_scheduler.config.prediction_type == "v_prediction":
                current_latent_estimate = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
            elif noise_scheduler.config.prediction_type == "epsilon":
                current_latent_estimate = (noisy_latents - beta_prod_t**0.5 * model_pred) / (alpha_prod_t**0.5)
            elif noise_scheduler.config.prediction_type == "sample":
                current_latent_estimate = model_pred
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            current_latent_estimate = current_latent_estimate / vae.config.scaling_factor
            current_estimate = decode_image(vae, current_latent_estimate, dtype=weight_dtype)

            # Normalize to [0, 1] then scale to [0, 255]
            current_estimate = (current_estimate + 1.0) / 2.0  # [-1, 1] → [0, 1]
            current_estimate = current_estimate * 255.0  # [0, 1] → [0, 255]
            
            ground_truth = batch["label"].to(device=accelerator.device, dtype=weight_dtype)
            
            # Convert to numpy for evaluation
            for i in range(current_estimate.shape[0]):
                pred_np = current_estimate[i].detach().to(torch.float32).cpu().numpy()
                target_np = ground_truth[i].detach().to(torch.float32).cpu().numpy()

                # Convert to grayscale if needed
                if pred_np.ndim == 3 and pred_np.shape[0] == 3:
                    pred_np = pred_np.mean(axis=0)
                if target_np.ndim == 3 and target_np.shape[0] == 3:
                    target_np = target_np.mean(axis=0)
                
                predictions.append(pred_np)
                targets.append(target_np)
    
    # Evaluate using IoU metric
    metrics = evaluator.evaluate_dataset(predictions, targets)
    
    # Return sample images for logging
    sample_predictions = predictions[:3] if len(predictions) >= 3 else predictions
    sample_targets = targets[:3] if len(targets) >= 3 else targets
    
    model.train()
    return metrics, sample_predictions, sample_targets

def main():
    args = parse_args()

    # Init accelerator and logger
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    if args.random_state_file is not None:
        logger.info(f"Loading random states from file {args.random_state_file}")
        random_states = torch.load(args.random_state_file)
        import random
        # Python's built-in random
        random.setstate(random_states["random_state"])
        # NumPy
        np.random.set_state(random_states["numpy_random_seed"])
        # Torch CPU
        torch.set_rng_state(random_states["torch_manual_seed"])
        # Torch CUDA (for each visible GPU)
        if "torch_cuda_manual_seed" in random_states:
            for i, cuda_state in enumerate(random_states["torch_cuda_manual_seed"]):
                if torch.cuda.device_count() > i:
                    torch.cuda.set_rng_state(cuda_state, device=i)

    # Save training arguments in a .txt file
    if accelerator.is_main_process:
        args_dict = vars(args)
        args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, 'w') as file:
            file.write(args_str)

    if args.noise_type is None:
        logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning (Stable Diffusion).")

    # Load model components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer       = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder    = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    unet           = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None)

    # Modify UNet input if noise_type is not None
    if args.noise_type is not None:
        if unet.config['in_channels'] != 8:
            replace_unet_conv_in(unet, repeat=2)
            logger.info("Unet conv_in layer replaced for (RGB + condition) input")

    # Freeze VAE and CLIP (text encoder), train only UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 has known issues on some GPUs. If you see problems, update to >=0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install xformers to use memory efficient attention.")

    # For saving/loading with accelerate >= 0.16.0
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_func      = IterExponential(
        total_iter_length=args.lr_total_iter_length * accelerator.num_processes,
        final_ratio=0.05,
        warmup_steps=args.lr_exp_warmup_steps * accelerator.num_processes
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Setup training dataset and dataloader
    if args.dataset_type == "rtm":
        if args.data_root is None:
            raise ValueError("--data_root is required for RTMDataset")
        
        train_dataset = RTMDataset(
            img_dir=os.path.join(args.data_root, args.img_dir),
            label_dir=os.path.join(args.data_root, args.label_dir),
            split_file=os.path.join(args.data_root, args.split_file),
            crop_size=(512, 512),
            mode='train',
            use_dct_quant=False,  # Disable DCT quantization for document forgery
            max_class_ratio=1.0,
            multiple_compressions=args.multiple_compressions,  # Use multiple compressions if specified
        )
        logger.info(f"Using RTMDataset with {len(train_dataset)} training samples")
        
    elif args.dataset_type == "doctamper":
        if args.lmdb_paths is None:
            raise ValueError("--lmdb_paths is required for DocTamperDataset")
        
        train_dataset = DocTamperDataset(
            roots=args.lmdb_paths,
            mode=args.dataset_mode,
            S=args.dataset_seed,
            T=8192,
            pilt=False,
            casia=False,
            ranger=1,
            max_nums=None,
            max_readers=64,
            multiple_compressions=args.multiple_compressions
        )
        logger.info(f"Using DocTamperDataset with {len(train_dataset)} training samples")
        
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None
    )

    # Setup validation dataset and dataloader
    val_dataloader = None
    if args.dataset_type == "rtm" and args.val_data_root is not None:
        val_dataset = RTMDataset(
            img_dir=os.path.join(args.val_data_root, args.val_img_dir),
            label_dir=os.path.join(args.val_data_root, args.val_label_dir),
            split_file=os.path.join(args.val_data_root, args.val_split_file),
            crop_size=(512, 512),
            mode='val',
            use_dct_quant=False,
            max_class_ratio=1.0
        )
        logger.info(f"Using validation dataset with {len(val_dataset)} samples")
    elif args.dataset_type == "doctamper" and args.val_lmdb_paths is not None:
        val_dataset = DocTamperDataset(
            roots=args.val_lmdb_paths,
            mode="val",
            S=args.dataset_seed,
            T=8192,
            pilt=False,
            casia=False,
            ranger=1,
            max_nums=None,
            max_readers=64
        )
        logger.info(f"Using validation dataset with {len(val_dataset)} samples")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None
    )

    # Prepare with accelerator (move to GPU, handle DDP, etc.)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Prepare validation dataloader if it exists
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Choose weight dtype for model
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        unet.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        unet.to(dtype=weight_dtype)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Compute number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps*accelerator.num_processes))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function to unwrap model if compiled
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Logging info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Dataset type = {args.dataset_type}")
    logger.info(f"  Num training examples = {len(train_dataset)}")
    if val_dataloader is not None:
        logger.info(f"  Num validation examples = {len(val_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if val_dataloader is not None:
        logger.info(f"  Validation steps = {args.validation_steps}")

    global_step = 0
    first_epoch = 0

    # Resume training if needed
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))  # sort by step
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Initialize loss function
    loss_fn = BinarySegmentationLoss()

    # Pre-compute empty text CLIP encoding
    empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token    = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)

    # For converting from predicted noise -> latents
    alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_prod  = 1 - alpha_prod

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Epoch {epoch} / {args.num_train_epochs}")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Encode RGB to latents
                rgb_latents = encode_image(
                    vae,
                    batch["image"].to(device=accelerator.device), dtype=weight_dtype
                )
                rgb_latents = rgb_latents * vae.config.scaling_factor

                # Timesteps
                timesteps = torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) \
                            * (noise_scheduler.config.num_train_timesteps - 1)
                timesteps = timesteps.long()
                noisy_latents = torch.zeros_like(rgb_latents)

                # UNet input: (rgb_latents, noisy_latents) if condition exists
                encoder_hidden_states = empty_encoding.repeat(len(batch["image"]), 1, 1)

                unet_input = rgb_latents

                model_pred = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False
                )[0]

                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
                beta_prod_t  = beta_prod[timesteps].view(-1, 1, 1, 1)

                if noise_scheduler.config.prediction_type == "v_prediction":
                    current_latent_estimate = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
                elif noise_scheduler.config.prediction_type == "epsilon":
                    current_latent_estimate = (noisy_latents - beta_prod_t**0.5 * model_pred) / (alpha_prod_t**0.5)
                elif noise_scheduler.config.prediction_type == "sample":
                    current_latent_estimate = model_pred
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                current_latent_estimate = current_latent_estimate / vae.config.scaling_factor
                current_estimate = decode_image(vae, current_latent_estimate, dtype=weight_dtype)

                # Normalize to [0, 1] then scale to [0, 255]
                current_estimate = (current_estimate + 1.0) / 2.0  # [-1, 1] → [0, 1]
                current_estimate = current_estimate * 255.0  # [0, 1] → [0, 255]
                
                ground_truth = batch["label"].to(device=accelerator.device, dtype=weight_dtype)
                if args.report_to == "wandb" and accelerator.is_main_process and global_step % args.wandb_image_interval == 0:
                    # Log the first image in the batch to wandb
                    img_to_log = current_estimate[0].detach().cpu().clamp(0, 255).to(torch.uint8)
                    gt_to_log = ground_truth[0].detach().cpu().clamp(0, 255).to(torch.uint8)
                    in_img_to_log = batch["image"][0].detach().cpu()
                    # If image has shape (C, H, W), convert to (H, W, C)
                    if img_to_log.ndim == 3 and img_to_log.shape[0] in [1, 3]:
                        img_to_log = img_to_log.permute(1, 2, 0)
                        # bin_img_to_log = img_to_log.numpy().mean(axis=-1) # Convert to single channel binary mask
                        # bin_img_to_log = (bin_img_to_log > 127.5)  # Convert to binary mask
                    if gt_to_log.ndim == 3 and gt_to_log.shape[0] in [1, 3]:
                        gt_to_log = gt_to_log.permute(1, 2, 0)
                    if in_img_to_log.ndim == 3 and in_img_to_log.shape[0] in [1, 3]:
                        in_img_to_log = in_img_to_log.permute(1, 2, 0)
                    wandb.log({
                        "predicted_image": wandb.Image(img_to_log.numpy(), caption="Predicted"),
                        "ground_truth": wandb.Image(gt_to_log.numpy(), caption="Ground Truth"),
                        "input_image": wandb.Image(in_img_to_log.numpy(), caption="Input Image"),
                        # "binary_mask": wandb.Image(bin_img_to_log, caption="Binary Mask"),
                        "global_step": global_step
                    }, step=global_step)

                # Compute loss
                estimation_loss = loss_fn(current_estimate, ground_truth)
                loss = loss + estimation_loss

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Detach and log
            avg_loss = accelerator.gather(loss.detach()).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # If we just finished an accumulation step...
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log the average step loss
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"white_loss": loss_fn.whiteLoss}, step=global_step)
                accelerator.log({"black_loss": loss_fn.blackLoss}, step=global_step)
                accelerator.log({"separation_loss": loss_fn.separationLoss}, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                # Run validation
                if val_dataloader is not None and global_step % args.validation_steps == 0:
                    logger.info(f"Running validation at step {global_step}")
                    val_metrics, sample_preds, sample_targets = evaluate_validation(
                        unwrap_model(unet), vae, val_dataloader, accelerator, weight_dtype, args
                    )
                    
                    # Log validation metrics to wandb
                    if args.report_to == "wandb" and accelerator.is_main_process:
                        # Log sample validation images
                        pred_img = sample_preds[0]
                        target_img = sample_targets[0]
                        
                        # Convert to uint8 for wandb
                        pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)
                        target_img = np.clip(target_img, 0, 255).astype(np.uint8)
                        pred_mask = (pred_img > 127.5).astype(np.uint8) * 255  # Convert to binary mask

                        wandb.log({
                            f"val_pred": wandb.Image(pred_img, caption=f"Validation Prediction"),
                            f"val_target": wandb.Image(target_img, caption=f"Validation Target"),
                            # f"val_pred_mask": wandb.Image(pred_mask, caption=f"Validation Prediction Mask"),
                            "global_step": global_step
                        }, step=global_step)
                    
                    # Log to accelerator
                    accelerator.log({
                        "val_iou": val_metrics["iou"],
                        "val_f1_score": val_metrics["f1_score"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_dice": val_metrics["dice"]
                    }, step=global_step)
                    
                    logger.info(f"Validation metrics at step {global_step}:")
                    logger.info(f"  IoU: {val_metrics['iou']:.4f}")
                    logger.info(f"  F1-Score: {val_metrics['f1_score']:.4f}")
                    logger.info(f"  Precision: {val_metrics['precision']:.4f}")
                    logger.info(f"  Recall: {val_metrics['recall']:.4f}")
                    logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                    logger.info(f"  Dice: {val_metrics['dice']:.4f}")

                # Checkpoint saving
                if global_step % args.checkpointing_steps == 0:
                    logger.info(f"Saving checkpoint at step {global_step}")
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            # Remove older checkpoints if exceeding limit
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)} oldest"
                                )
                                logger.info(f"Removing: {', '.join(removing_checkpoints)}")
                                for rm_ckpt in removing_checkpoints:
                                    rm_ckpt_path = os.path.join(args.output_dir, rm_ckpt)
                                    shutil.rmtree(rm_ckpt_path)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # Show step loss in progress bar
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # Delete / free memory
            del rgb_latents, noisy_latents, model_pred
            if 'current_latent_estimate' in locals():
                del current_latent_estimate
            if 'current_estimate' in locals():
                del current_estimate
            if 'ground_truth' in locals():
                del ground_truth
            del loss

            # Early stopping
            if global_step >= args.max_train_steps:
                break

        # End of epoch

    # Post-training: create pipeline and save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            timestep_spacing="trailing",
            revision=args.revision,
            variant=args.variant
        )
        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            variant=args.variant,
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)

    logger.info("Finished training.")
    accelerator.end_training()

if __name__ == "__main__":
    main() 