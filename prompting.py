# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

#############################################
# Gaussian Heatmap Generator
#############################################
def create_gaussian_heatmap(H, W, point, sigma, device='cpu'):
    """
    Creates a Gaussian heatmap of size (H, W) centered at the given normalized point.
    The heatmap is scaled from [0,1] to [-1,1] and returned with an extra channel.
    """
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
    # point is (normalized_y, normalized_x)
    dist_sq = (x_grid - point[1])**2 + (y_grid - point[0])**2
    heatmap = torch.exp(-dist_sq / (2 * sigma**2))
    heatmap = heatmap * 2 - 1  # scale from [0,1] -> [-1,1]
    return heatmap.unsqueeze(0)  # shape: (1, H, W)

#############################################
# Bilateral Solver (Refinement) Function
#############################################
def refine_with_bilateral_solver(sim_map, guidance_img, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Refines a similarity map using cv2.ximgproc.jointBilateralFilter with the guidance image.
    """
    if not (hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'jointBilateralFilter')):
        print("WARNING: cv2.ximgproc.jointBilateralFilter is not available. Install opencv-contrib-python.")
        print("Skipping refinement.")
        return sim_map # Return unrefined map if filter is not available

    sim_map_8u = np.clip(sim_map * 255, 0, 255).astype(np.uint8)
    refined = cv2.ximgproc.jointBilateralFilter(guidance_img, sim_map_8u, d, sigmaColor, sigmaSpace)
    refined_float = refined.astype(np.float32) / 255.0
    return refined_float

##################################
# Process a Single Prompt Point
##################################
def generate_mask_for_single_prompt(feature_image_path, prompt_point_xy, output_mask_path,
                                   gaussian_sigma=0.01, manual_threshold=3, epsilon=1e-6):
    """
    Generates and saves a binary mask for a single prompt point on a feature image.
    The image is processed at its original dimensions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the feature image
    try:
        img = Image.open(feature_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Feature image not found at {feature_image_path}")
        return
    except Exception as e:
        print(f"Error opening image {feature_image_path}: {e}")
        return

    W_orig, H_orig = img.size # Get original dimensions

    img_np = np.array(img).astype(np.float32)
    features_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)  # (3, H, W)
    _, H, W = features_tensor.shape # H, W will be H_orig, W_orig

    # Create guidance image for bilateral filter (original RGB image)
    guidance_img_np = np.array(img).astype(np.uint8) # cv2 expects uint8 BGR or grayscale
    if guidance_img_np.shape[2] == 3: # RGB
        guidance_img_bgr = cv2.cvtColor(guidance_img_np, cv2.COLOR_RGB2BGR)
    else: # Grayscale
        guidance_img_bgr = guidance_img_np


    print(f"Image dimensions: H={H}, W={W}")
    print(f"Prompt (x,y): {prompt_point_xy}")

    # 2. Normalize prompt point and create Gaussian heatmap
    # prompt_point_xy is (x, y), create_gaussian_heatmap expects (norm_y, norm_x)
    # Ensure prompt point is within image bounds
    if not (0 <= prompt_point_xy[0] < W and 0 <= prompt_point_xy[1] < H):
        print(f"Error: Prompt point ({prompt_point_xy[0]},{prompt_point_xy[1]}) is outside image bounds ({W-1},{H-1}).")
        return

    norm_y = prompt_point_xy[1] / (H - 1) if H > 1 else 0.5
    norm_x = prompt_point_xy[0] / (W - 1) if W > 1 else 0.5
    norm_point = (norm_y, norm_x)

    prompt_heatmap = create_gaussian_heatmap(H, W, norm_point, sigma=gaussian_sigma, device=device)
    prompt_weights = (prompt_heatmap + 1) / 2  # Convert from [-1,1] to [0,1].

    # 3. Compute weighted color and similarity map
    weighted_sum_rgb = torch.sum(features_tensor * prompt_weights, dim=(1, 2))
    total_weight = torch.sum(prompt_weights)
    query_color_rgb = weighted_sum_rgb / (total_weight + epsilon)

    diff_rgb = features_tensor - query_color_rgb[:, None, None]
    distance_map_rgb = torch.norm(diff_rgb, dim=0)
    similarity_map_rgb = 1.0 / (distance_map_rgb + epsilon)

    min_val_rgb = similarity_map_rgb.min()
    max_val_rgb = similarity_map_rgb.max()
    # Add epsilon to prevent division by zero if max_val_rgb == min_val_rgb
    normalized_similarity_rgb = (similarity_map_rgb - min_val_rgb) / (max_val_rgb - min_val_rgb + epsilon)
    normalized_similarity_rgb = normalized_similarity_rgb.view(H, W)

    # 4. Refine similarity map
    # Use guidance_img_bgr which is the original color image in BGR uint8 format
    refined_sim_map_np = refine_with_bilateral_solver(
        normalized_similarity_rgb.cpu().numpy().astype(np.float32), guidance_img_bgr
    )

    # 5. Threshold to produce binary mask
    binary_mask = ((refined_sim_map_np * 255) > manual_threshold).astype(np.uint8) * 255

    # 6. Save the binary mask
    try:
        cv2.imwrite(output_mask_path, binary_mask)
        print(f"Successfully saved binary mask to {output_mask_path}")
    except Exception as e:
        print(f"Error saving mask to {output_mask_path}: {e}")

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a binary mask for a single prompt point.")
    parser.add_argument("--feature_image", type=str, required=True,
                        help="Path to the input feature PNG file.")
    parser.add_argument("--prompt_x", type=int, required=True,
                        help="X-coordinate of the prompt point.")
    parser.add_argument("--prompt_y", type=int, required=True,
                        help="Y-coordinate of the prompt point.")
    parser.add_argument("--output_mask", type=str, default="mask.png",
                        help="Path to save the output binary mask PNG file.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Gaussian sigma for the heatmap.")
    parser.add_argument("--threshold", type=int, default=3,
                        help="Manual threshold for binarization (applied to 0-255 scale map).")

    args = parser.parse_args()

    # Check if cv2.ximgproc is available early
    if not (hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'jointBilateralFilter')):
        print("Warning: opencv-contrib-python might not be installed or cv2.ximgproc is not found.")
        print("The 'refine_with_bilateral_solver' function will skip refinement if the filter is unavailable.")
        print("To install: pip install opencv-contrib-python")


    generate_mask_for_single_prompt(
        feature_image_path=args.feature_image,
        prompt_point_xy=(args.prompt_x, args.prompt_y),
        output_mask_path=args.output_mask,
        gaussian_sigma=args.sigma,
        manual_threshold=args.threshold
    )