# gen2seg official inference pipeline code for Stable Diffusion model
# 
# This code was adapted from Marigold and Diffusion E2E Finetuning. 
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper, along with the two works above. 

import tempfile
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
import jpegio

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

    def __call__(self, rgb_image, inst_image):
        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            inst_image = self.horizontal_flip(inst_image)
        rgb_image   = self.resize(rgb_image)
        inst_image = self.resize_nn(inst_image)
        if self.crop != None:
            rgb_image, inst_image = self.random_crop_torch(rgb_image, inst_image, self.crop)

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

from albumentations.pytorch import ToTensorV2

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
            instance_tensor = torch.from_numpy(np.array(instance_image))*2.0-1.0

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


class RTMDataset(Dataset):
    def __init__(self, img_dir, label_dir, split_file, crop_size=(512, 512), mode='train', use_dct_quant=False, max_class_ratio=0.9, multiple_compressions=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        self.mode = mode
        self.use_dct_quant = use_dct_quant
        self.max_class_ratio = max_class_ratio
        self.multiple_compressions = multiple_compressions
        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] → [-1,1]
        ])

        with open(split_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image, label = self._load_sample(sample)
        
        # Convert label to tensor
        label = self.totsr(image=label.copy())['image']
        
        # Apply augmentations in training mode
        if self.mode == 'train':
            if random.uniform(0, 1) < 0.5:
                image = self.hflip(image)
                label = self.hflip(label)
            if random.uniform(0, 1) < 0.5:
                image = self.vflip(image)
                label = self.vflip(label)

            if self.multiple_compressions:
                # Apply multiple JPEG compressions with different qualities
                q = random.randint(75,100)
                q2 = random.randint(75,100)
                q3 = random.randint(75,100)
                
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    choicei = random.randint(0,2)
                    if choicei>1:
                        image.save(tmp.name,"JPEG",quality=q3)
                        image=Image.open(tmp.name)
                    if choicei>0:
                        image.save(tmp.name,"JPEG",quality=q2)
                        image=Image.open(tmp.name)
                    image.save(tmp.name,"JPEG",quality=q)
                    image = Image.open(tmp.name)

        # Extract Y quantization table from the JPEG image if enabled
        if self.use_dct_quant:
            qtb, dct = self._get_Y_quantization_DCT_table(sample)
            if qtb is None:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))
        else:
            qtb, dct = None, None
        
        # Apply cropping
        if self.use_dct_quant:
            if self.mode == 'train':
                # Apply random crop with maximum class ratio of 0.75
                image, dct, label = self._random_crop(image, dct, label)
            else:
                image, dct, label = self._crop(image, dct, label)
                
            if image is None or label is None or dct is None:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))
        else:
            if self.mode == 'train':
                image, label = self._random_crop_simple(image, label)
            else:
                image, label = self._crop_simple(image, label)
                
            if image is None or label is None:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        result = {
            'image': self.toctsr(image),
            'label': label.float()
        }
        
        if self.use_dct_quant:
            result['rgb'] = np.clip(np.abs(dct), 0, 20)
            result['q'] = qtb
        return result
    def _load_sample(self, s):
        """Load a single sample image and its corresponding binary label using PIL."""
        img_path = f"{self.img_dir}/{s}.jpg"
        label_path = f"{self.label_dir}/{s}.png"

        # image = Image.open(img_path).convert("L")
        image = Image.open(img_path)
        image = image.convert("RGB")  # Convert to RGB for consistency
        
        label = Image.open(label_path).convert("L")
        label = label.convert("RGB")
        
        # Convert to numpy array and ensure values are either 0 or 255
        label_array = np.array(label)
        # Convert to binary: any non-zero value becomes 255
        label_binary = (label_array > 0).astype(np.uint8) * 255
        
        return image, label_binary

    def _get_class_ratio(self, label, ignore_index=255):
        """Return a dict of class index -> class ratio, excluding ignore_index."""
        # Convert tensor to numpy if needed
        if torch.is_tensor(label):
            label = label.numpy()
        
        # Handle 3-channel RGB labels by converting to binary mask
        if len(label.shape) == 3 and label.shape[0] == 3:
            # Convert RGB to binary: [255, 255, 255] -> 1, [0, 0, 0] -> 0
            binary_label = (label[0, :, :] > 127).astype(np.uint8)
        else:
            binary_label = label
        
        unique, counts = np.unique(binary_label, return_counts=True)

        # Filter out ignored class
        valid_mask = unique != ignore_index
        unique = unique[valid_mask]
        counts = counts[valid_mask]

        if counts.sum() == 0:
            return {}  # all pixels were ignored

        class_ratios = dict(zip(unique, counts / counts.sum()))
        return class_ratios

    def _get_Y_quantization_DCT_table(self, sample):
        """Extract the Y quantization table and DCT coefficients from the JPEG image."""
        img_path = f"{self.img_dir}/{sample}.jpg"
        jpg = jpegio.read(img_path)
        if (hasattr(jpg, 'quant_tables') and len(jpg.quant_tables) > 0 and 
            hasattr(jpg, 'coef_arrays') and len(jpg.coef_arrays) > 0):
            qtable = torch.from_numpy(jpg.quant_tables[0]).to(dtype=torch.int64)
            dct = jpg.coef_arrays[0].copy()
            return qtable, dct
        else:
            return None, None
    def _random_crop(self, image, dct, label, stride=8, max_tries=20):
        """Apply random crop to image, DCT coefficients, and label with class ratio constraints."""
        h, w = image.size[1], image.size[0]
        crop_h, crop_w = self.crop_size[1], self.crop_size[0]
        
        if h < crop_h or w < crop_w:
            return None, None, None

        max_x = w - crop_w
        max_y = h - crop_h
        if max_x < 0 or max_y < 0:
            return None, None, None

        possible_x = list(range(0, max_x + 1, stride))
        possible_y = list(range(0, max_y + 1, stride))
        if not possible_x or not possible_y:
            return None, None, None

        for _ in range(max_tries):
            x = random.choice(possible_x)
            y = random.choice(possible_y)

            cropped_image = image.crop((x, y, x + crop_w, y + crop_h))
            # Handle 3-channel RGB labels
            if len(label.shape) == 3 and label.shape[0] == 3:
                cropped_label = label[:, y:y + crop_h, x:x + crop_w]
            else:
                cropped_label = label[y:y + crop_h, x:x + crop_w]

            cropped_dct = dct[y:y + crop_h, x:x + crop_w]

            if self._ensure_max_class_ratio(cropped_label, ignore_index=255) is not None:
                return cropped_image, cropped_dct, cropped_label

        return None, None, None
    
    def _crop(self, image, dct, label):
        """Crop the image, DCT coefficients, and label to the specified size."""
        h, w = image.size[1], image.size[0]
        if h < self.crop_size[1] or w < self.crop_size[0]:
            return None, None, None
            
        x = (w - self.crop_size[0]) // 2
        y = (h - self.crop_size[1]) // 2
        
        cropped_image = image.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
        
        # Handle 3-channel RGB labels
        if len(label.shape) == 3 and label.shape[0] == 3:
            cropped_label = label[:, y:y + self.crop_size[1], x:x + self.crop_size[0]]
        else:
            cropped_label = label[y:y + self.crop_size[1], x:x + self.crop_size[0]]
        
        cropped_dct = dct[y:y + self.crop_size[1], x:x + self.crop_size[0]]
        
        return cropped_image, cropped_dct, cropped_label
    
    def _random_crop_simple(self, image, label, stride=8, max_tries=20):
        """Apply random crop to image and label without DCT coefficients."""
        h, w = image.size[1], image.size[0]
        crop_h, crop_w = self.crop_size[1], self.crop_size[0]

        if h < crop_h or w < crop_w:
            return None, None

        max_x = w - crop_w
        max_y = h - crop_h
        if max_x < 0 or max_y < 0:
            return None, None

        possible_x = list(range(0, max_x + 1, stride))
        possible_y = list(range(0, max_y + 1, stride))
        if not possible_x or not possible_y:
            return None, None

        for _ in range(max_tries):
            x = random.choice(possible_x)
            y = random.choice(possible_y)

            cropped_image = image.crop((x, y, x + crop_w, y + crop_h))
            # Handle 3-channel RGB labels
            if len(label.shape) == 3 and label.shape[0] == 3:
                cropped_label = label[:, y:y + crop_h, x:x + crop_w]
            else:
                cropped_label = label[y:y + crop_h, x:x + crop_w]

            if self._ensure_max_class_ratio(cropped_label, ignore_index=255) is not None:
                return cropped_image, cropped_label

        return None, None
    
    def _crop_simple(self, image, label):
        """Crop the image and label to the specified size without DCT coefficients."""
        h, w = image.size[1], image.size[0]
        if h < self.crop_size[1] or w < self.crop_size[0]:
            return None, None
            
        x = (w - self.crop_size[0]) // 2
        y = (h - self.crop_size[1]) // 2
        
        cropped_image = image.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
        
        # Handle 3-channel RGB labels
        if len(label.shape) == 3 and label.shape[0] == 3:
            cropped_label = label[:, y:y + self.crop_size[1], x:x + self.crop_size[0]]
        else:
            cropped_label = label[y:y + self.crop_size[1], x:x + self.crop_size[0]]
        
        return cropped_image, cropped_label
    
    def _ensure_max_class_ratio(self, label, ignore_index=255):
        """Ensure that the label contains more than one class (excluding ignore_index)
        and that the dominant class does not exceed the class property ratio."""
        class_ratios = self._get_class_ratio(label, ignore_index)

        if len(class_ratios) <= 1:  # Only one class (or all ignored)
            return None

        max_class_ratio = max(class_ratios.values())
        if max_class_ratio >= self.max_class_ratio:
            return None

        return label

import lmdb
import six

class DocTamperDataset(Dataset):
    def __init__(self, roots, mode, S, T=8192, pilt=False, casia=False, ranger=1, max_nums=None, max_readers=64, multiple_compressions=True):
        self.cnts = []
        self.lens = []
        self.envs = []
        self.pilt = pilt
        self.casia = casia
        self.ranger = ranger
        
        # Initialize LMDB environments
        for root in roots:
            if '$' in root:
                root_use, nums = root.split('$')
                nums = int(nums)
                self.envs.append(lmdb.open(root_use, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False))
                with self.envs[-1].begin(write=False) as txn:
                    str = 'num-samples'.encode('utf-8')
                    nSamples = int(txn.get(str))
                    if max_nums is not None:
                        nSamples = min(nSamples, max_nums)
                self.lens.append(nSamples * nums)
                self.cnts.append(nSamples)
            else:
                self.envs.append(lmdb.open(root, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False))
                with self.envs[-1].begin(write=False) as txn:
                    str = 'num-samples'.encode('utf-8')
                    nSamples = int(txn.get(str))
                    if max_nums is not None:
                        nSamples = min(nSamples, max_nums)
                self.lens.append(nSamples)
                self.cnts.append(nSamples)
        
        self.lens = np.array(self.lens)
        self.sums = np.cumsum(self.lens)
        self.len_sum = len(self.sums)
        self.nSamples = self.lens.sum()
        self.S = S
        self.T = T
        self.mode = mode
        self.multiple_compressions = multiple_compressions
        
        print('*' * 60)
        print('Dataset initialized!', S, pilt)
        print('*' * 60)
        
        # Initialize random indices
        npr = np.arange(self.nSamples)
        np.random.seed(S)
        self.idxs = np.random.choice(self.nSamples, self.nSamples, replace=False)
        
        # Initialize transforms
        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] → [-1,1]
        ])

    def calnum(self, num):
        if num < self.lens[0]:
            return 0, num % self.cnts[0]
        else:
            for li, l in enumerate(self.sums):
                if ((l <= num) and ((li == self.len_sum) or (num < self.sums[li + 1]))):
                    return (li + 1), ((num - l) % self.cnts[li + 1])

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, idx):
        itm_num = self.idxs[idx]
        env_num, index = self.calnum(itm_num)
        
        with self.envs[env_num].begin(write=False) as txn:
            # Load image
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            # Load label
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)
            info_key = 'info-%09d' % index
            infobuf = txn.get(info_key.encode('utf-8'))

            
            H, W = mask.shape
            if (H != 512) or (W != 512):
                return self.__getitem__(random.randint(0, self.nSamples - 1))
            
            # Data augmentation
            if random.uniform(0, 1) < 0.5:
                im = im.rotate(90)
                mask = np.rot90(mask, 1)
            
            mask = self.totsr(image=mask.copy())['image']
            
            if random.uniform(0, 1) < 0.5:
                im = self.hflip(im)
                mask = self.hflip(mask)
            
            if random.uniform(0, 1) < 0.5:
                im = self.vflip(im)
                mask = self.vflip(mask)
                
            if self.multiple_compressions and self.mode == 'train':
                # Apply multiple JPEG compressions with different qualities
                q = random.randint(75,100)
                q2 = random.randint(75,100)
                q3 = random.randint(75,100)
                elaQuality = 80
                
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    if infobuf and '1' in infobuf:
                        choicei=0
                    else:
                        choicei = random.randint(0,2)
                    if choicei>1:
                        im.save(tmp.name,"JPEG",quality=q3)
                        im=Image.open(tmp.name)
                    if choicei>0:
                        im.save(tmp.name,"JPEG",quality=q2)
                        im=Image.open(tmp.name)
                    im.save(tmp.name,"JPEG",quality=q)
                    im = Image.open(tmp.name)
                    if self.use_dct_quant:
                        jpgImage = jpegio.read(tmp.name)
                        dct = jpgImage.coef_arrays[0].copy()
                    if self.use_dct_quant:
                        oldimg = np.array(im)
                        im.save(tmp.name, "JPEG", quality=elaQuality)
                        newimg = Image.open(tmp.name)
                        # newimg = newimg.convert('RGB')
                        newimg = np.array(newimg)
                        ela = np.abs(newimg - oldimg)
                        ela = np.clip(ela, 0, 255).astype(np.uint8)

            # Convert to grayscale RGB and apply normalization
            im = im.convert('RGB')
            mask = mask.squeeze(0)
            mask = torch.from_numpy(np.stack([mask * 255] * 3, axis=0).astype(np.uint8))
            # print(f"mask shape: {mask.shape}")
            # print(f"mask shape: {mask.shape}")
            # print(f"im shape: {np.array(im).shape}")
            
            result = {
                'image': self.toctsr(im),
                'label': mask.float(),
            }
            if self.use_dct_quant:
                if dct is not None:
                    result['dct'] = np.clip(np.abs(dct), 0, 20)
                if ela is not None:
                    result['ela'] = np.clip(ela, 0, 255)
                else:
                    result['ela'] = None
            
            return result


if __name__ == "__main__":
    dataset = DocTamperDataset(["/netscratch/hussain/TamperText/Datasets/DocTamperV1/DocTamperV1-TestingSet"], mode='train', S=0, T=8192, pilt=False, casia=False, ranger=1, max_nums=None, max_readers=64, multiple_compressions=True)
    
    for i in range(10):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image min/max: {sample['image'].min():.3f}/{sample['image'].max():.3f}")
        print(f"  Label shape: {sample['label'].shape}")
        print(f"  Label min/max: {sample['label'].min():.3f}/{sample['label'].max():.3f}")
        print(f"  Label unique values: {torch.unique(sample['label'])}")
        if "dct" in dataset.replace_channels:
            print(f"  DCT shape: {sample['rgb'].shape}")
            print(f"  DCT min/max: {sample['rgb'].min():.3f}/{sample['rgb'].max():.3f}")
        if "ela" in dataset.replace_channels:
            print(f"  ELA shape: {sample['ela'].shape}")
            print(f"  ELA min/max: {sample['ela'].min():.3f}/{sample['ela'].max():.3f}")
        print()