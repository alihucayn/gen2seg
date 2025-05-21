#!/usr/bin/env python
"""
Single Image AutomaskGen Script for SAM (Segment Anything Model)
with masks overlaid on a black background.

This script:
  - Loads a pre-trained SAM model from a checkpoint.
  - Creates an automatic mask generator using SAM.
  - Processes a single input image to generate segmentation masks.
  - Overlays each mask (with a unique random color) on a black background.
  - Saves the resulting image.
    
User Parameters:
  - INPUT_IMAGE: Path to the input image.
  - OUTPUT_IMAGE: Path where the output image will be saved.
  - MODEL_TYPE: Type of SAM model to use (e.g., "vit_h", "vit_l", "vit_b").
  - CHECKPOINT_PATH: Path to the SAM model checkpoint.
  - DEVICE: Computation device ("cuda" or "cpu").
"""

import os
import cv2
import numpy as np
import random

# Import SAM model components. Adjust the import based on your SAM repository installation.
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#########################################
# USER PARAMETERS (edit these variables)
#########################################
INPUT_IMAGE = "FILL THIS IN"                                              # Path to the input image.
OUTPUT_IMAGE = "sam.png"                                            # Path where the output image will be saved.
MODEL_TYPE = "vit_h"                                                # Options typically include "vit_h", "vit_l", or "vit_b".
CHECKPOINT_PATH = "PATH TO sam_vit_h_4b8939.pth"  # Path to the SAM model checkpoint.
DEVICE = "cuda"                                                     # Device for inference ("cuda" or "cpu").

print("Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)

# Instantiate the automatic mask generator with default parameters.
mask_generator = SamAutomaticMaskGenerator(sam)

# Load the input image using OpenCV.
image_bgr = cv2.imread(INPUT_IMAGE)
if image_bgr is None:
    print(f"Error: Unable to load image at {INPUT_IMAGE}")
    exit()
# Convert the image to RGB since SAM expects RGB images.
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Create a black background with the same size as the input image.
black_background = np.zeros_like(image_rgb)

print("Generating masks...")
# Generate masks automatically.
masks = mask_generator.generate(image_rgb)
num_masks = len(masks)
print(f"Generated {num_masks} masks.")

# Pre-generate a unique random color for each mask
unique_colors = []
for _ in range(num_masks):
    # generate a random RGB tuple in [5,250]
    color = tuple(random.randint(5, 250) for _ in range(3))
    # ensure uniqueness
    while color in unique_colors:
        color = tuple(random.randint(5, 250) for _ in range(3))
    unique_colors.append(color)

# Overlay each mask onto the black background with its unique color.
for mask, color in zip(masks, unique_colors):
    segmentation = mask["segmentation"]  # boolean mask
    color_arr = np.array(color, dtype=np.uint8)
    black_background[segmentation] = color_arr

# Convert the resulting image from RGB to BGR for saving via OpenCV.
output_bgr = cv2.cvtColor(black_background, cv2.COLOR_RGB2BGR)

cv2.imwrite(OUTPUT_IMAGE, output_bgr)
print(f"Output image saved to: {OUTPUT_IMAGE}")

