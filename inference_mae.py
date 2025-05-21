# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper, along with the two works above. 


import os
import time
import torch
from gen2seg_mae_pipeline import gen2segMAEInstancePipeline  # Custom pipeline for MAE
from transformers import ViTMAEForPreTraining, ViTMAEConfig, AutoImageProcessor
from safetensors.torch import load_file
from PIL import Image
import numpy as np

def infer_single_image(image_path: str, output_path: str, model_path: str, device: str = "cuda:0"):
    print(f"Loading MAE pipeline on {device} for single image inference...")
    
    # Load the MAE model manually using safetensors.
    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "model.safetensors")
    config = ViTMAEConfig.from_json_file(config_path)
    state_dict = load_file(weights_path)
    model = ViTMAEForPreTraining(config)
    model.load_state_dict(state_dict)
    
    # Load the image processor (using a pretrained processor from facebook/vit-mae-huge).
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-huge")
    config.size = 224  # Ensure the image size matches the expected dimensions.
    
    # Instantiate the pipeline and move it to the desired device.
    pipe = gen2segMAEInstancePipeline(model=model, image_processor=image_processor).to(device)
    
    # Load the image, storing the original size, then resize for inference.
    try:
        orig_image = Image.open(image_path).convert("RGB")
        orig_size = orig_image.size  # (width, height)
        image = orig_image.resize((224, 224))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Run inference.
    start_time = time.time()
    with torch.no_grad():
        pipe_output = pipe([image])
        prediction = pipe_output.prediction[0]
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")
    
    # Convert the prediction to an image.
    seg = np.array(prediction.squeeze()).astype(np.uint8)
    seg_img = Image.fromarray(seg)
    
    # Resize the segmentation output back to the original image size.
    seg_img = seg_img.resize(orig_size, Image.LANCZOS)
    
    # Save the output image.
    seg_img.save(output_path)
    print(f"Saved output image to {output_path}")

if __name__ == "__main__":
    # Example usage: Update these paths as needed.
    image_path = "lionking-crop.png"       # Path to the input image.
    output_path = "seg_mae.png"     # Path to save the output image.
    model_path = "/nfs_share3/om/diffusion-e2e-ft/model-finetuned/mae_full_e2e_ft_mean_mixed_sqrtsep_meansep_norm"
    device = "cuda:0"  # Change to "cpu" if no GPU is available.

    infer_single_image(image_path, output_path, model_path, device)
