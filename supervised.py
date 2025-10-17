#!/usr/bin/env python3
"""
Multi-GPU DDP Supervised Training Script - Baseline for MaskJEPA Comparison

Uses: SwinV2-Small backbone + Same segmentation head as fine_tune.py

Usage:
    # Single GPU:
    python supervised.py
    
    # Multi-GPU (automatic detection):
    torchrun --nproc_per_node=auto supervised.py
    
    # Specific number of GPUs:
    torchrun --nproc_per_node=4 supervised.py
"""

# CRITICAL: Set NCCL environment variables BEFORE importing torch
import os
os.environ['NCCL_TIMEOUT'] = '7200'              # 2 hours
os.environ['NCCL_BLOCKING_WAIT'] = '1'           # Synchronous error handling
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'    # Better error reporting
os.environ['NCCL_IB_DISABLE'] = '1'              # Disable InfiniBand
os.environ['NCCL_SOCKET_NTHREADS'] = '4'         # Reduce overhead
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'        # Reduce overhead

# NOW import torch and everything else
import gc, math, numpy as np, atexit, csv, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import timm
from model import UPerNetHead

from Dataloader import batch_size_downstream
from Dataset import ADE20KDataset, ade_collate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import lr_lambda

# DDP Setup
def setup_ddp():
    """Initialize DDP and return local rank, world size, and device"""
    # Check if we're running with torchrun (has all required env vars)
    if all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']):
        # Running with torchrun - full DDP setup
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        
        # Set device for this rank
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        
        return rank, world_size, device
    else:
        # Check available GPUs
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to single CPU")
            return 0, 1, "cpu"
        
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s): {[f'GPU {i}' for i in range(gpu_count)]}")
        
        if gpu_count > 1:
            print(f"You have {gpu_count} GPUs! For multi-GPU training, run:")
            print(f"   torchrun --nproc_per_node={gpu_count} supervised.py")
            print("Falling back to single GPU training for now...")
        
        return 0, 1, "cuda:0"

def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()

# Initialize DDP
rank, world_size, device = setup_ddp()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main_process = rank == 0

# Register cleanup
atexit.register(cleanup_ddp)

# Create DDP-compatible dataloaders
ade_train_dataset = ADE20KDataset(split="training")
ade_val_dataset = ADE20KDataset(split="validation")

if world_size > 1:
    train_sampler = DistributedSampler(ade_train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(ade_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        ade_train_dataset,
        batch_size=batch_size_downstream,
        sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=ade_collate
    )
    val_loader = DataLoader(
        ade_val_dataset,
        batch_size=batch_size_downstream,
        sampler=val_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=ade_collate
    )
else:
    train_loader = DataLoader(
        ade_train_dataset,
        batch_size=batch_size_downstream,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=ade_collate
    )
    val_loader = DataLoader(
        ade_val_dataset,
        batch_size=batch_size_downstream,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ade_collate
    )

if is_main_process:
    print(f"Train dataset: {len(ade_train_dataset):,} samples")
    print(f"Val dataset: {len(ade_val_dataset):,} samples")
    print(f"Train DataLoader: {len(train_loader)} batches per GPU")
    print(f"Val DataLoader: {len(val_loader)} batches per GPU")
    if world_size > 1:
        print(f"Using DistributedSampler for {world_size} GPUs")

IGNORE_INDEX = 255
NUM_CLASSES = 150

# -----------------------
# Supervised Segmentation Model (uses SegmentationHead from model.py)
# -----------------------

class SupervisedSegmentationModel(nn.Module):
    """
    SUPERVISED BASELINE with UPerNet Head
    
    Uses: SwinV2-Small backbone + UPerNet (proven architecture)
    - Uses all 4 feature stages for multi-scale fusion
    - Proven segmentation architecture for best performance
    """
    def __init__(self, num_classes=150):
        super().__init__()
        # Load pretrained SwinV2-Small backbone with ALL stages
        self.backbone = timm.create_model(
            "swin_small_patch4_window7_224", 
            pretrained=True,
            img_size=512,               # Increased to 512 for better resolution
            num_classes=0, 
            global_pool="",
            features_only=True,
            out_indices=[0, 1, 2, 3],  # Extract ALL 4 stages for UPerNet
            strict_img_size=False,     # Allow dynamic padding
            dynamic_img_pad=True       # Enable dynamic padding
        )
        
        # Get feature dimensions from all stages
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)  # Match the img_size
            features = self.backbone(dummy_input)
            # SwinV2-Small channels: [96, 192, 384, 768] typically
            backbone_channels = [f.shape[1] for f in features]
        
        # UPerNet head (proven architecture) - MAXIMUM CAPACITY FOR 40%+ mIoU
        self.upernet_head = UPerNetHead(
            in_channels_list=backbone_channels,
            num_classes=num_classes,
            fpn_out_channels=256,  # QUADRUPLED for maximum capacity
            ppm_out_channels=512,  # QUADRUPLED for maximum capacity
            dropout=0.05  # MINIMAL dropout for maximum learning
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Extract multi-scale backbone features
        features = self.backbone(x)  # List of 4 feature maps
        
        # Generate segmentation logits using UPerNet
        logits = self.upernet_head(features, (H, W))
        
        return logits

# ADE20K label mapping is now done in Dataset.py

class StreamingSegMetrics:
    """GPU-side streaming confusion matrix for mIoU (no CPU stalls)."""
    def __init__(self, num_classes, ignore_index=255, device=None):
        self.C = num_classes
        self.ignore = ignore_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = torch.zeros((self.C, self.C), dtype=torch.float64, device=self.device)
    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1)
        tgt  = target
        valid = (tgt != self.ignore)
        if valid.any():
            pred = pred[valid]
            tgt  = tgt[valid]
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
        iou  = torch.where(denom_iou > 0, diag / denom_iou, torch.nan)
        miou  = torch.nanmean(iou).item()
        return miou
    def reset(self):
        self.conf.zero_()

def visualize_segmentation(images, true_masks, logits, epoch, save_path, num_samples=4):
    pred = logits.argmax(1)
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 12))
    mean = torch.tensor([0.485,0.456,0.406], device=images.device).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=images.device).view(3,1,1)
    for i in range(min(num_samples, images.size(0))):
        img = torch.clamp(images[i]*std + mean, 0, 1).permute(1,2,0).cpu().numpy()
        axes[0,i].imshow(img); axes[0,i].set_title(f"Original {i+1}"); axes[0,i].axis('off')
        gt = true_masks[i].cpu().numpy(); gt_rgb = np.zeros((*gt.shape,3))
        pr = pred[i].cpu().numpy();      pr_rgb = np.zeros((*pr.shape,3))
        for cls in range(NUM_CLASSES):
            m1 = (gt==cls); m2 = (pr==cls)
            if m1.any(): gt_rgb[m1] = plt.cm.tab20(cls%20)[:3]
            if m2.any(): pr_rgb[m2] = plt.cm.tab20(cls%20)[:3]
        axes[1,i].imshow(gt_rgb); axes[1,i].set_title(f"Ground Truth {i+1}"); axes[1,i].axis('off')
        axes[2,i].imshow(pr_rgb); axes[2,i].set_title(f"Prediction {i+1}"); axes[2,i].axis('off')
    plt.suptitle(f"Epoch {epoch} - Supervised Segmentation Results", fontsize=16)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

# -----------------------
# Model Setup
# -----------------------
if is_main_process:
    print("Creating supervised baseline model (SwinV2-Small + UPerNet)...")
    
model = SupervisedSegmentationModel(num_classes=NUM_CLASSES).to(device)  # UPerNet for best performance
model = model.to(memory_format=torch.channels_last)

# Fix BatchNorm for multi-GPU training
if world_size > 1:
    # Convert BatchNorm to SyncBatchNorm for proper multi-GPU training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if is_main_process:
        print("Converted BatchNorm to SyncBatchNorm for multi-GPU training")
    
    # Wrap with DDP - FIXED DEVICE IDS
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if is_main_process:
        print(f"Model wrapped with DDP across {world_size} GPUs")
else:
    if is_main_process:
        print("Using single GPU training")

# -----------------------
# Training Configuration - FIXED FOR FINETUNING
# -----------------------
num_epochs = 300
print_every = 100

# Split learning rates for backbone vs head
backbone_lr = 5e-4 # if shit go back to 5e-4
head_lr = 5e-3 # if shit go back to 5e-3      
weight_decay = 0.01
warmup_epochs = 0 

# Early stopping config
early_stop_patience = 20
best_val_miou = 0.0
epochs_no_improve = 0

# Split parameters for different learning rates
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'backbone' in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

# Create optimizer with different learning rates
optimizer = AdamW([
    {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
    {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
])

# Learning rate scheduler with warmup
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: lr_lambda(epoch, num_epochs, warmup_epochs, min_lr_ratio=0.01)
)

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
scaler = GradScaler('cuda')

save_dir = "./supervised_training_output"
os.makedirs(save_dir, exist_ok=True)

# CSV logging setup (rank 0 only)
csv_path = os.path.join(save_dir, "supervised_log.csv")
if is_main_process:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'epoch_time_min', 'train_ce', 'train_miou', 'val_ce', 'val_miou', 'lr'])

if is_main_process:
    print("Starting supervised baseline training...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training setup
train_ce_hist, val_ce_hist, val_miou_hist = [], [], []

# Access base model for saving
base_model = model.module if hasattr(model, 'module') else model

# Performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Better performance

# NCCL TIMEOUT FIXES
os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes instead of 10
os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Enable blocking wait
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Better error handling

# -----------------------
# Training Loop
# -----------------------
# Track training time
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    # Set epoch for DistributedSampler
    if world_size > 1:
        train_loader.sampler.set_epoch(epoch)
        
    model.train()
    epoch_ce = 0.0
    epoch_meter = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)
    print_meter = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)

    for batch_idx, batch in enumerate(train_loader):
        images = batch["images"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        masks = batch["masks"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(images)
            loss = criterion(logits, masks)

        # Update metrics (every 100 batches for speed)
        with torch.no_grad():
            if (batch_idx % 100) == 0:
                epoch_meter.update(logits, masks)
                print_meter.update(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_ce += loss.item()

        if (batch_idx % print_every) == 0 and is_main_process:
            miou_b = print_meter.get()
            print(f"  Batch {batch_idx:4d}/{len(train_loader)} | CE: {loss.item():.4f} | mIoU: {miou_b:.4f}")
            print_meter.reset()

        del images, masks, logits, loss

    epoch_ce /= len(train_loader)
    train_miou = epoch_meter.get()  # Get training metrics
    train_ce_hist.append(epoch_ce)
    scheduler.step()

    # Validation every epoch
    model.eval()
    val_ce = 0.0
    val_meter = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = batch["masks"].to(device, non_blocking=True)
            
            with autocast('cuda', enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(images)
                ce = criterion(logits, masks)
                
            val_ce += ce.item()
            val_meter.update(logits, masks)
            del images, masks, logits, ce

    val_ce /= len(val_loader)
    val_miou = val_meter.get()
    val_ce_hist.append(val_ce)
    val_miou_hist.append(val_miou)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    if is_main_process:
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train CE: {epoch_ce:.4f} mIoU: {train_miou:.4f} | Val CE: {val_ce:.4f} mIoU: {val_miou:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{epoch_duration/60:.2f}", f"{epoch_ce:.4f}", f"{train_miou:.4f}",
                           f"{val_ce:.4f}", f"{val_miou:.4f}", f"{scheduler.get_last_lr()[0]:.2e}"])

    # Early stopping and save best by validation mIoU
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        epochs_no_improve = 0
        if is_main_process:
            best_path = os.path.join(save_dir, "best_supervised_model.pt")
            torch.save({
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_miou': val_miou,
                'val_miou': val_miou
            }, best_path)
            print(f"  New best Val mIoU: {best_val_miou:.4f}! Saved -> {best_path}")
    else:
        epochs_no_improve += 1
        if is_main_process:
            print(f"  [Early Stop] No improvement ({epochs_no_improve}/{early_stop_patience})")
        if epochs_no_improve >= early_stop_patience:
            if is_main_process:
                print(f"  [Early Stop] Patience exceeded. Stopping supervised training.")
            break

    # Visualizations (every 20 epochs for speed)
    if (epoch + 1) % 20 == 0 and is_main_process:
        with torch.no_grad():
            vis_batch = next(iter(val_loader))
            vis_images = vis_batch["images"].to(device).to(memory_format=torch.channels_last)
            vis_masks = vis_batch["masks"].to(device)
            with autocast('cuda', enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                vis_logits = model(vis_images)
            vis_path = os.path.join(save_dir, f"supervised_epoch_{epoch+1:03d}.png")
            visualize_segmentation(vis_images, vis_masks, vis_logits, epoch+1, vis_path)
            print(f"  Saved visualization: {vis_path}")
            del vis_batch, vis_images, vis_masks, vis_logits

    torch.cuda.empty_cache()
    gc.collect()

if is_main_process:
    total_training_time = time.time() - training_start_time
    print("Supervised training completed!")
    print(f"Best Val mIoU: {best_val_miou:.4f}")
    print(f"Total supervised training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    
    # Log total time
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TOTAL', f"{total_training_time/60:.1f}", '', '', '', '', ''])

# Save final model
if is_main_process:
    final_path = os.path.join(save_dir, "final_supervised_model.pt")
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'train_ce_losses': train_ce_hist,
        'val_ce_losses': val_ce_hist,
        'val_mious': val_miou_hist,
        'best_val_miou': best_val_miou
    }, final_path)
    print(f"Final supervised model saved: {final_path}")

# Cleanup DDP
cleanup_ddp()
