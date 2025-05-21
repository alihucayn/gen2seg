# gen2seg official inference pipeline code for Stable Diffusion model
#
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 



import torch
from diffusers.utils import load_image
from gen2seg_sd_pipeline import gen2segSDPipeline  # Import your custom pipeline
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2


# Load the image
image_path =  '/nfs_share3/om/diffusion-e2e-ft/19.png'
image = Image.open(image_path).convert("RGB")#.rotate(180)
orig_res = image.size

# Define the path to your custom checkpoint
custom_checkpoint_path = "model-finetuned/stable_diffusion_e2e_ft_instance_fixed_meansep" #"model-finetuned/stable_diffusion_e2e_ft_instance_bbox"
# Load the pipeline and generate the depth map
with torch.no_grad():
    pipe = gen2segSDPipeline.from_pretrained(
        custom_checkpoint_path,
        # torch_dtype=torch.float16,  # Use float16 for better performance
        use_safetensors=True,         # Use safetensors if available
        # device_map="auto"           # Uncomment for automatic device mapping
    ).to("cuda")  # Ensure the pipeline is moved to CUDA

    # Generate depth map
    seg = pipe(image).prediction.squeeze()
    seg = np.array(seg).astype(np.uint8)

    # Ensure the image is RGB
    """if seg.ndim == 2:  # If grayscale, convert to RGB
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)"""

    Image.fromarray(seg).resize(orig_res, Image.LANCZOS).save("seg.png")
