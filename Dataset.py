import os
import glob
import numpy as np
import torch
import random
from PIL import Image
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode as IM
from math import ceil

# Reproducibility
seed = 1337
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

IMG_SIZE = 512  # Updated for SwinV2-Small optimal size

# =============================================================================
# JEPA ImageNet Dataset for Pretraining
# =============================================================================
class JEPADataset(Dataset):
    def __init__(self, root="./imagenet-100", split="train", img_size=IMG_SIZE):
        self.dataset = load_from_disk(root)[split]
        self.img_size = img_size
        # Image transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load and transform image
        img = self.dataset[idx]["image"].convert("RGB")
        img_tensor = self.transform(img)
        return {
            "image": img_tensor
        }

# =============================================================================
# ADE20K Dataset for Segmentation
# =============================================================================
class ADE20KDataset(Dataset):
    def __init__(self, root="ADEChallengeData2016", split="training", img_size=IMG_SIZE):
        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        self.img_size = img_size
        self.split = split
        self.items = []
        
        # Find image-mask pairs
        for img_path in glob.glob(os.path.join(self.img_dir, "*.jpg")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            ann_path = os.path.join(self.ann_dir, stem + ".png")
            if os.path.exists(ann_path):
                self.items.append((img_path, ann_path))
        self.items.sort()
        
        # DIFFERENT TRANSFORMS PER SPLIT
        if split == "training":
            # Training: Manual transforms in __getitem__ (no self.transform needed)
            self.transform = None
        else:
            # Validation: NO random transforms - deterministic only
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),  # Just resize
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, ann_path = self.items[idx]
        
        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path)
        
        if self.split == "training":
            # AGGRESSIVE TRAINING AUGMENTATION
            # Much more aggressive ADE20K augmentation for 40%+ mIoU
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, 
                scale=(0.3, 2.5),  # MUCH WIDER range - zoom out more and zoom in more aggressively
                ratio=(0.5, 2.0)  # Much more variety in aspect ratios
            )
            img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size), interpolation=IM.BILINEAR, antialias=True)
            mask = TF.resized_crop(mask, i, j, h, w, (self.img_size, self.img_size), interpolation=IM.NEAREST)
            
            # Apply same flip to both
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            # Much more aggressive rotation
            if random.random() < 0.5:  # 50% chance - much more rotation
                angle = random.uniform(-30, 30)  # Max ±30° - much more aggressive
                img = TF.rotate(img, angle, interpolation=IM.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=IM.NEAREST, fill=255)
            
            # Convert to tensor and normalize
            img = TF.to_tensor(img)
            
            # Much stronger color augmentation
            if random.random() < 0.9:  # 90% chance - almost always apply
                img = transforms.ColorJitter(
                    brightness=0.4,  # Stronger
                    contrast=0.4,    # Stronger
                    saturation=0.4,  # Stronger
                    hue=0.15         # Stronger hue
                )(img)
            
            # Random blur - 20% chance
            if random.random() < 0.2:
                kernel_size = random.choice([3, 5, 7])
                img = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size])
            
            # Random gamma correction - 20% chance
            if random.random() < 0.2:
                gamma = random.uniform(0.7, 1.5)
                img = TF.adjust_gamma(img, gamma)
            
            # Random grayscale
            if random.random() < 0.1:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)
            
            img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            # VALIDATION: DETERMINISTIC TRANSFORMS - No randomness
            img = self.transform(img)  # Just resize + normalize
            
            # Mask: Just resize (no random transforms)
            mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        
        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask, dtype="int64"))
        
        # ADE20K label mapping: 0=background, 1-150=classes → 0-149=classes, 255=ignore
        mask = torch.where((mask == 0) | (mask > 150), 255, mask - 1).long()
        
        return img, mask

def jepa_collate(batch):
    images = torch.stack([item["image"] for item in batch])
    return {"images": images}

def ade_collate(batch):
    imgs, masks = zip(*batch)
    return {"images": torch.stack(imgs), "masks": torch.stack(masks)}