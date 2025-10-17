#!/usr/bin/env python3
"""
Quick ADE20K Test Script - Overfit + Validate Model Setup

Tests:
- ADE20K dataset loading and preprocessing
- SwinV2-Small + UPerNet model overfitting (50 epochs on 1-2 images)
- Expect ~0.8+ Dice/mIoU after overfitting if setup is correct
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import timm

from Dataset import ADE20KDataset, ade_collate
from model import UPerNetHead

# Constants from supervised.py
IMG_SIZE = 512
NUM_CLASSES = 150
IGNORE_INDEX = 255

# ADE20K label mapping is now done in Dataset.py

class StreamingSegMetrics:
    """GPU-side streaming confusion matrix for mIoU"""
    def __init__(self, num_classes, ignore_index=255, device=None):
        self.C = num_classes
        self.ignore = ignore_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = torch.zeros((self.C, self.C), dtype=torch.float64, device=self.device)
    
    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1)
        tgt = target
        valid = (tgt != self.ignore)
        if valid.any():
            pred = pred[valid]
            tgt = tgt[valid]
            idx = tgt * self.C + pred
            bins = torch.bincount(idx, minlength=self.C*self.C).reshape(self.C, self.C).to(self.conf.dtype)
            self.conf += bins
    
    @torch.no_grad()
    def get(self):
        h = self.conf
        diag = torch.diag(h)
        sum_row = h.sum(1)
        sum_col = h.sum(0)
        denom_iou = sum_row + sum_col - diag
        iou = torch.where(denom_iou > 0, diag / denom_iou, torch.nan)
        miou = torch.nanmean(iou).item()
        return miou

class TestModel(nn.Module):
    """SwinV2-Small + UPerNet for testing"""
    def __init__(self, num_classes=150):
        super().__init__()
        # Load pretrained SwinV2-Small backbone with ALL stages
        self.backbone = timm.create_model(
            "swin_small_patch4_window7_224", 
            pretrained=True,
            img_size=IMG_SIZE,
            num_classes=0, 
            global_pool="",
            features_only=True,
            out_indices=[0, 1, 2, 3],  # Extract ALL 4 stages for UPerNet
            strict_img_size=False,
            dynamic_img_pad=True
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            features = self.backbone(dummy_input)
            backbone_channels = [f.shape[1] for f in features]
        
        # UPerNet head
        self.upernet_head = UPerNetHead(
            in_channels_list=backbone_channels,
            num_classes=num_classes,
            fpn_out_channels=256,
            ppm_out_channels=512,
            dropout=0.1
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone(x)  # List of 4 feature maps
        logits = self.upernet_head(features, (H, W))
        return logits

def visualize_test(images, true_masks, logits, save_path="test_results.png"):
    """Visualize test results with Original, GT, and Prediction side by side"""
    pred = logits.argmax(1)
    num_images = min(2, images.size(0))
    
    # Create 3 columns: Original, Ground Truth, Prediction
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize images - ensure same device
    device = images.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    
    # Create color map for classes
    colors = plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES))
    colors = colors[:, :3]  # Take only RGB, not RGBA
    
    for i in range(num_images):
        # 1. Original Image
        img = torch.clamp(images[i] * std + mean, 0, 1).permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original Image {i+1}", fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')
        
        # 2. Ground Truth Segmentation
        gt = true_masks[i].cpu().numpy()
        gt_rgb = np.zeros((*gt.shape, 3))
        
        # Color each class
        for cls in range(NUM_CLASSES):
            mask = (gt == cls)
            if mask.any():
                gt_rgb[mask] = colors[cls % len(colors)]
        
        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title(f"Ground Truth {i+1}", fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')
        
        # 3. Predicted Segmentation
        pr = pred[i].cpu().numpy()
        pr_rgb = np.zeros((*pr.shape, 3))
        
        # Color each class (same colors as GT)
        for cls in range(NUM_CLASSES):
            mask = (pr == cls)
            if mask.any():
                pr_rgb[mask] = colors[cls % len(colors)]
        
        axes[i, 2].imshow(pr_rgb)
        axes[i, 2].set_title(f"Prediction {i+1}", fontsize=14, fontweight='bold')
        axes[i, 2].axis('off')
    
    # Add overall title
    plt.suptitle("ADE20K Segmentation Results: Original → Ground Truth → Prediction", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add column headers
    fig.text(0.17, 0.95, "Original", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.95, "Ground Truth", ha='center', fontsize=12, fontweight='bold')
    fig.text(0.83, 0.95, "Prediction", ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Enhanced visualization saved: {save_path}")

def get_different_images(train_loader, device, image_indices=[10, 25, 50, 100]):
    """Get specific images from the dataset by index"""
    print(f"Selecting images at indices: {image_indices}")
    
    # Get batches and select images
    all_images = []
    all_masks = []
    
    batch_count = 0
    for batch in train_loader:
        batch_images = batch["images"]
        batch_masks = batch["masks"]
        
        for i in range(batch_images.shape[0]):
            if batch_count * train_loader.batch_size + i in image_indices:
                all_images.append(batch_images[i])
                all_masks.append(batch_masks[i])
                print(f"  Selected image {batch_count * train_loader.batch_size + i}")
        
        batch_count += 1
        
        # Stop when we have enough images
        if len(all_images) >= len(image_indices):
            break
    
    # Stack into tensors
    images = torch.stack(all_images).to(device)
    masks = torch.stack(all_masks).to(device)
    
    print(f"Selected {images.shape[0]} images for overfitting")
    return images, masks

def overfit_on_few_images(model, train_loader, device, num_epochs=50, image_indices=[10, 25, 50, 100]):
    """Overfit model on specific images for many epochs to test setup"""
    print(f"OVERFITTING on specific images for {num_epochs} epochs...")
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)  # Higher LR for faster overfitting
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = GradScaler('cuda')
    
    # Get images to overfit on
    images, masks = get_different_images(train_loader, device, image_indices)
    
    print(f"Overfitting on {images.shape[0]} images for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(images)
            loss = criterion(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pred = logits.argmax(1)
                # Quick mIoU calculation
                valid_mask = masks != IGNORE_INDEX
                if valid_mask.any():
                    pred_valid = pred[valid_mask]
                    gt_valid = masks[valid_mask]
                    correct = (pred_valid == gt_valid).float().mean()
                    print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss = {loss.item():.4f}, Accuracy = {correct.item():.4f}")
    
    print("✅ Overfitting completed!")
    # Return training data for testing
    return model, images, masks

def main():
    print("🧪 ADE20K Quick Test - Overfit + Validate Model Setup")
    print("=" * 60)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load training dataset
    print("Loading ADE20K training dataset...")
    train_dataset = ADE20KDataset(split="training", img_size=IMG_SIZE)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True, 
        num_workers=0,
        collate_fn=ade_collate
    )
    
    # Load test dataset (validation split - no random transforms)
    print("Loading ADE20K validation dataset...")
    test_dataset = ADE20KDataset(split="validation", img_size=IMG_SIZE)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2,  # Just 2 images for quick test
        shuffle=False, 
        num_workers=0,
        collate_fn=ade_collate
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Load model
    print("Loading SwinV2-Small + UPerNet model...")
    model = TestModel(num_classes=NUM_CLASSES).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # OVERFIT THE MODEL on specific images for many epochs
    # Try different image indices - you can change these numbers!
    # Examples: [1, 2, 3, 4] or [100, 200, 300, 400] or [50, 150, 250, 350]
    image_indices = [20, 45, 80, 120]
    model, train_images, train_masks = overfit_on_few_images(model, train_loader, device, num_epochs=200, image_indices=image_indices)
    
    # Switch to eval mode
    model.eval()
    
    # Test on the SAME images we trained on (to verify overfitting worked)
    print("Running inference on the SAME images we trained on...")
    images = train_images
    masks = train_masks
    
    print(f"Input shapes: images {images.shape}, masks {masks.shape}")
    
    # Run inference
    with torch.no_grad():
        logits = model(images)
        print(f"Output logits shape: {logits.shape}")
        
        print(f"Masks shape: {masks.shape}")
        print(f"Mask value range: {masks.min().item()} to {masks.max().item()}")
        print(f"Ignore pixels: {(masks == IGNORE_INDEX).sum().item()}")
        
        # Calculate metrics
        metrics = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device)
        metrics.update(logits, masks)
        miou = metrics.get()
        
        print(f"\nRESULTS:")
        print(f"mIoU: {miou:.4f}")
        
        # Expected results check (after overfitting)
        if miou > 0.8:
            print("✅ PERFECT: Overfitting worked - setup is excellent!")
        elif miou > 0.6:
            print("✅ GOOD: Strong overfitting - setup is working well!")
        elif miou > 0.4:
            print("⚠️  MEDIUM: Some overfitting, but could be better")
        else:
            print("❌ LOW: Overfitting failed - check labels/ignore handling!")
            print("   After 50 epochs on 2 images, should see mIoU > 0.6")
        
        # Visualize results
        print("Generating visualization...")
        visualize_test(images, masks, logits)
        
        # Additional diagnostics
        print(f"\nDIAGNOSTICS:")
        valid_gt = masks[masks != IGNORE_INDEX]
        pred_classes = logits.argmax(1)
        valid_pred = pred_classes[masks != IGNORE_INDEX]
        
        print(f"Valid GT pixels: {valid_gt.numel()}")
        print(f"Unique classes in GT: {torch.unique(valid_gt).numel()}")
        print(f"GT class distribution: {torch.bincount(valid_gt.flatten()).nonzero().flatten()[:10]}")  # First 10 classes
        
        print(f"Unique classes in Pred: {torch.unique(valid_pred).numel()}")
        print(f"Pred class distribution: {torch.bincount(valid_pred.flatten()).nonzero().flatten()[:10]}")  # First 10 classes
        
        print(f"Prediction confidence: {torch.softmax(logits, dim=1).max().item():.4f}")
        print(f"Most common pred class: {torch.mode(valid_pred.flatten())[0].item()}")
        print(f"Most common GT class: {torch.mode(valid_gt.flatten())[0].item()}")
        
        # Check if model predicts one class
        if torch.unique(valid_pred).numel() == 1:
            print("⚠️  WARNING: Model predicting only one class - likely untrained!")
        elif torch.unique(valid_pred).numel() < 5:
            print("⚠️  WARNING: Model predicting very few classes - may need training")

if __name__ == "__main__":
    main()
