# gen2seg official inference pipeline code for Stable Diffusion model
# 
# This code was adapted from Marigold and Diffusion E2E Finetuning. 
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper, along with the two works above. 

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2
import re
from pathlib import Path

import torchvision.transforms.functional as TF

#################
# Dataset Mixer
#################

class MixedDataLoader:
    def __init__(self, loader1, loader2, split1=9, split2=1):
        self.loader1 = loader1
        self.loader2 = loader2
        self.split1 = split1
        self.split2 = split2
        self.frac1, self.frac2 = self.get_split_fractions()
        self.randchoice1=None

    def __iter__(self):
        self.loader_iter1 = iter(self.loader1)
        self.loader_iter2 = iter(self.loader2)
        self.randchoice1 = self.create_split()
        self.indx = 0
        return self
    
    def get_split_fractions(self):
        size1 = len(self.loader1)
        size2 = len(self.loader2)
        effective_fraction1 = min((size2/size1) * (self.split1/self.split2), 1) 
        effective_fraction2 = min((size1/size2) * (self.split2/self.split1), 1) 
        print("Effective fraction for loader1: ", effective_fraction1)
        print("Effective fraction for loader2: ", effective_fraction2)
        return effective_fraction1, effective_fraction2

    def create_split(self):
        randchoice1 = [True]*int(len(self.loader1)*self.frac1) + [False]*int(len(self.loader2)*self.frac2)
        np.random.shuffle(randchoice1)
        return randchoice1

    def __next__(self):
        if self.indx == len(self.randchoice1):
            raise StopIteration
        if self.randchoice1[self.indx]:
            self.indx += 1
            return next(self.loader_iter1)
        else:
            self.indx += 1
            return next(self.loader_iter2)
        
    def __len__(self):
        return int(len(self.loader1)*self.frac1) + int(len(self.loader2)*self.frac2)
    

#################
# Transforms 
#################

# Hyperism
class SynchronizedTransform_Hyper:
    def __init__(self, H, W, crop=None):
        self.resize          = transforms.Resize((H,W))
        self.resize_nn    = transforms.Resize((H,W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor       = transforms.ToTensor()
        self.crop = crop
    def random_crop_torch(self, image, instance_mask, crop_size, max_attempts=1000):
        """
        Applies the same random crop to the image and its masks using PyTorch.
        Ensures that none of the cropped images are entirely black.

        Parameters:
        - image: PIL.Image or torch.Tensor, the original image.
        - instance_mask: PIL.Image or torch.Tensor, the instance segmentation mask.
        - crop_size: tuple (height, width), the size of the crop.
        - max_attempts: int, maximum number of crop attempts to find a valid crop.

        Returns:
        - Tuple of cropped (image, instance_mask).

        Raises:
        - ValueError: If a valid crop is not found within the maximum number of attempts.
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif len(crop_size) != 2:
            raise ValueError("crop_size should be int or a tuple of (height, width)")

        # Get dimensions
        if isinstance(image, torch.Tensor):
            _, height, width = image.shape
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise TypeError("Unsupported image type. Must be PIL.Image or torch.Tensor.")

        crop_height, crop_width = crop_size

        if width < crop_width or height < crop_height:
            raise ValueError("Crop size should be smaller than the image size.")

        def is_all_black(img):
            """
            Checks if the given image or tensor is entirely black.

            Parameters:
            - img: PIL.Image or torch.Tensor

            Returns:
            - bool: True if all pixels are zero, False otherwise.
            """
            if isinstance(img, torch.Tensor):
                return torch.all(img == 0).item()
            elif isinstance(img, Image.Image):
                img_tensor = TF.to_tensor(img)
                return torch.all(img_tensor == 0).item()
            else:
                raise TypeError("Unsupported image type for checking blackness.")

        attempt = 0
        for attempt in range(max_attempts):
            attempt += 1

            # Randomly select top-left corner
            if isinstance(image, torch.Tensor):
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
            else:  # PIL.Image
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)

            # Apply crop to all images
            if isinstance(image, torch.Tensor):
                cropped_image = TF.crop(image, top, left, crop_height, crop_width)
                cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)
            else:  # PIL.Image
                cropped_image = TF.crop(image, top, left, crop_height, crop_width)
                cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)

            # Check if any of the cropped images are all black
            if not (is_all_black(cropped_image) or is_all_black(cropped_instance)):
                return cropped_image, cropped_instance
        print("rand not found hypersim")
    # If a valid crop wasn't found after max_attempts
        # Randomly select top-left corner
        if isinstance(image, torch.Tensor):
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
        else:  # PIL.Image
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)

        # Apply crop to all images
        if isinstance(image, torch.Tensor):
            cropped_image = TF.crop(image, top, left, crop_height, crop_width)
            cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)
        else:  # PIL.Image
            cropped_image = TF.crop(image, top, left, crop_height, crop_width)
            cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)
        return cropped_image, cropped_instance

    def __call__(self, rgb_image, seg_image, inst_image):
        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            seg_image = self.horizontal_flip(seg_image)
            inst_image = self.horizontal_flip(inst_image)
        rgb_image   = self.resize(rgb_image)
        seg_image = self.resize_nn(seg_image)
        inst_image = self.resize_nn(inst_image)
        if self.crop != None:
            rgb_image, inst_image, seg_image = self.random_crop_torch(rgb_image, inst_image, seg_image, self.crop)

        # to tensor
        rgb_tensor = self.to_tensor(rgb_image)*2.0-1.0
        inst_tensor = self.to_tensor(inst_image)*255

        return rgb_tensor, inst_tensor

import math
# Virtual KITTI 2
class SynchronizedTransform_VKITTI:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    # KITTI benchmark crop from Marigold:
    # https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/dataset/kitti_dataset.py#L75
    @staticmethod
    def kitti_benchmark_crop(input_img):
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216
        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out

    def random_crop_torch(self, image, instance_mask, crop_size, max_attempts=1000):
        """
        Applies the same random crop to the image and its masks using PyTorch.
        Ensures that none of the cropped images are entirely black.

        Parameters:
        - image: PIL.Image or torch.Tensor, the original image.
        - instance_mask: PIL.Image or torch.Tensor, the instance segmentation mask.
        - crop_size: tuple (height, width), the size of the crop.
        - max_attempts: int, maximum number of crop attempts to find a valid crop.

        Returns:
        - Tuple of cropped (image, instance_mask).

        Raises:
        - ValueError: If a valid crop is not found within the maximum number of attempts.
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif len(crop_size) != 2:
            raise ValueError("crop_size should be int or a tuple of (height, width)")

        # Get dimensions
        if isinstance(image, torch.Tensor):
            _, height, width = image.shape
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise TypeError("Unsupported image type. Must be PIL.Image or torch.Tensor.")

        crop_height, crop_width = crop_size

        # If the image is smaller than the crop size, resize it so that the smallest side is at least as big as the crop size.
        if width < crop_width or height < crop_height:
            scale_factor = max(crop_width / width, crop_height / height)
            new_width = int(math.ceil(width * scale_factor))
            new_height = int(math.ceil(height * scale_factor))
            if isinstance(image, torch.Tensor):
                image = TF.resize(image, [new_height, new_width], interpolation=TF.InterpolationMode.BILINEAR)
                instance_mask = TF.resize(instance_mask, [new_height, new_width], interpolation=TF.InterpolationMode.NEAREST)
            else:  # PIL.Image
                image = TF.resize(image, (new_height, new_width), interpolation=Image.BILINEAR)
                instance_mask = TF.resize(instance_mask, (new_height, new_width), interpolation=Image.NEAREST)
            width, height = new_width, new_height

        def is_all_black(img):
            """
            Checks if the given image or tensor is entirely black.

            Parameters:
            - img: PIL.Image or torch.Tensor

            Returns:
            - bool: True if all pixels are zero, False otherwise.
            """
            if isinstance(img, torch.Tensor):
                return torch.all(img == 0).item()
            elif isinstance(img, Image.Image):
                img_tensor = TF.to_tensor(img)
                return torch.all(img_tensor == 0).item()
            else:
                raise TypeError("Unsupported image type for checking blackness.")

        # Try to find a valid crop that is not completely black
        for attempt in range(max_attempts):
            # Randomly select top-left corner
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)

            # Apply crop to all images
            if isinstance(image, torch.Tensor):
                cropped_image = TF.crop(image, top, left, crop_height, crop_width)
                cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)
            else:  # PIL.Image
                cropped_image = TF.crop(image, top, left, crop_height, crop_width)
                cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)

            # Check if any of the cropped images are all black
            if not (is_all_black(cropped_image) or is_all_black(cropped_instance)):
                return cropped_image, cropped_instance

        print("rand not found kitti")
        # If a valid crop wasn't found after max_attempts, perform one final crop
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        if isinstance(image, torch.Tensor):
            cropped_image = TF.crop(image, top, left, crop_height, crop_width)
            cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)
        else:
            cropped_image = TF.crop(image, top, left, crop_height, crop_width)
            cropped_instance = TF.crop(instance_mask, top, left, crop_height, crop_width)
        return cropped_image, cropped_instance

    def __call__(self, rgb_image, instance, res):  # orig 368x1024
        # Horizontal flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            instance = self.horizontal_flip(instance)

        rgb_image, instance = self.random_crop_torch(rgb_image, instance, res)

        # Convert to tensor
        rgb_tensor = self.to_tensor(rgb_image) * 2.0 - 1.0    
        instance = self.to_tensor(instance) * 255


        return rgb_tensor, instance

#####################
# Training Datasets
#####################

from tqdm import tqdm
import glob
# Hypersim   
import os
import glob
import re
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class Hypersim(Dataset):
    def __init__(self, root_dir, transform=True, height=480, width=640, crop=None):
        self.root_dir = root_dir
        self.pairs = self._find_pairs()
        self.transform = SynchronizedTransform_Hyper(H=height, W=width, crop=crop) if transform else None

    @staticmethod
    def get_frame_number(filename):
        # Extracts the frame number from filenames like frame.0002.rgb.jpg
        match = re.search(r'frame\.(\d+)', str(filename))
        return int(match.group(1)) if match else -1

    def _find_pairs(self):
        pairs = []
       # print("IMPORTANT, USING INPAINTED DATASET")
        rgb_base = os.path.join(self.root_dir, "rgb")
        instance_base = os.path.join(self.root_dir, "instance-rgb")
        # Find all RGB images
        rgb_files = sorted(glob.glob(os.path.join(rgb_base, "**", "*.jpg"), recursive=True))
        print(f"####################################################{len(rgb_files)}")

        for rgb_path in tqdm(rgb_files, desc="Collecting RGB-Semantic pairs"):
            frame_num = self.get_frame_number(rgb_path)
            if frame_num == -1:
                continue

            # Extract scene_id and camera name from path
            path_parts = rgb_path.split(os.sep)
            scene_id = path_parts[-4]  # e.g., <scene_id>
            camera_dir_name = path_parts[-2].replace("final", "geometry")  # scene_cam_03_geometry_preview

            camera_dir_name = path_parts[-2].replace("final_preview", "geometry_hdf5")
            instance_dir = os.path.join(instance_base, scene_id, "images", camera_dir_name)
            
            instance_path = os.path.join(instance_dir, f"frame.{str(frame_num).zfill(4)}.png")

            if os.path.exists(instance_path):
                pairs.append({
                    "rgb_path": rgb_path,
                    "instance_path": instance_path #FIXXX
                })

        print(f"Hypersim size: {len(pairs)}")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        rgb_image = Image.open(pair['rgb_path']).convert('RGB')
        instance_image = Image.open(pair['instance_path'])

        if self.transform:
            rgb_tensor, instance_tensor = self.transform(rgb_image, instance_image)
        else:
            print("no transform")
            rgb_tensor = transforms.ToTensor()(rgb_image) *2.0 -1.0 # [-1,1]
            instance_tensor = torch.from_numpy(np.semantic_tensorarray(instance_image))*2.0-1.0

        return {
            "rgb": rgb_tensor,
            "instance": instance_tensor,
            "no_bg": False
        }



class VirtualKITTI2(Dataset):
    def __init__(self, root_dir, res, transform=True):
        self.root_dir = root_dir
        self.transform = SynchronizedTransform_VKITTI() if transform else None
        self.pairs = self._find_pairs()
        self.res = res

    def _find_pairs(self):
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
        cameras = ["Camera_0", "Camera_1"]

        vkitti2_rgb_path = self.root_dir

        pairs = []
        for scene in scenes:
            for weather in weather_conditions:
                for camera in cameras:
                    rgb_dir = os.path.join(vkitti2_rgb_path, scene, weather, "frames", "rgb", camera)
                    instance_dir = os.path.join(vkitti2_rgb_path, scene, weather, "frames", "instanceSegmentation", camera)

                    if os.path.exists(rgb_dir) and os.path.exists(instance_dir):
                        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
                        for rgb_file in rgb_files:
                            instance_file = rgb_file.replace("rgb", "instancegt").replace(".jpg", ".png")


                            rgb_path = os.path.join(rgb_dir, rgb_file)
                            instance_path = os.path.join(instance_dir, instance_file)
                            if os.path.exists(instance_path):
                                pairs.append((rgb_path, instance_path))

                    else:
                        print("##########################################################not found")
                        print(rgb_dir)
                        print(instance_dir)
                        exit(0)
        print(f"VKitti2 size: {len(pairs)}")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, instance_path = self.pairs[idx]

        # Load RGB
        rgb_image = Image.open(rgb_path).convert('RGB')
        instance_img = Image.open(instance_path).convert('RGB') 

        if self.transform is not None:
            rgb_tensor, instance_tensor = self.transform(rgb_image, instance_img, self.res)
        else:
            print("no transform")

            rgb_tensor = transforms.ToTensor()(rgb_image)*2.0 - 1.0
            instance_tensor = torch.from_numpy(np.array(instance_img))*2.0-1.0

        return {
            "rgb": rgb_tensor,
            "instance": instance_tensor,
            "no_bg": True
        }
