#!/usr/bin/env python3
"""
Multi-GPU DDP Fine-tuning Script for MaskJEPA

Usage:
    # Single GPU:
    python fine_tune.py
    
    # Multi-GPU (automatic detection):
    torchrun --nproc_per_node=auto fine_tune.py
    
    # Specific number of GPUs:
    torchrun --nproc_per_node=4 fine_tune.py
"""

# CRITICAL: Set NCCL environment variables BEFORE importing torch
import os
os.environ['NCCL_TIMEOUT'] = '14400'             # 4 hours (was timing out at 10 min)
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

from MaskJEPA import MaskJEPA2D
from model import SegmentationHead
from Dataloader import batch_size_downstream
from Dataset import ADE20KDataset, ade_collate
from torch.utils.data import DataLoader
from torch.utils.data import Subset
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
        
        # Initialize process group with explicit timeout
        from datetime import timedelta
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=14400))
        
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
            print(f"   torchrun --nproc_per_node={gpu_count} fine_tune.py")
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

# Quick test configuration
QUICK_TEST = True  # Set to True for quick test with limited data

# Create DDP-compatible dataloaders
ade_train_dataset = ADE20KDataset(split="training")
ade_val_dataset = ADE20KDataset(split="validation")

# Quick test mode - limit datasets to small subsets
if QUICK_TEST:
    ade_train_dataset = Subset(ade_train_dataset, range(min(500, len(ade_train_dataset))))
    ade_val_dataset = Subset(ade_val_dataset, range(min(100, len(ade_val_dataset))))
    if is_main_process:
        print(f"QUICK TEST MODE: Limited train dataset to {len(ade_train_dataset)} samples")
        print(f"QUICK TEST MODE: Limited val dataset to {len(ade_val_dataset)} samples")

if world_size > 1:
    train_sampler = DistributedSampler(ade_train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(ade_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    downstream_train_loader = DataLoader(
        ade_train_dataset,
        batch_size=batch_size_downstream,
        sampler=train_sampler,
        num_workers=1,  # Reduced from 8 to avoid worker overload
        pin_memory=True,
        collate_fn=ade_collate,
        persistent_workers=False  # Disabled - causes hangs in multi-GPU training
    )
    downstream_val_loader = DataLoader(
        ade_val_dataset,
        batch_size=batch_size_downstream,
        sampler=val_sampler,
        num_workers=1,  # Reduced from 8 to avoid worker overload
        pin_memory=True,
        collate_fn=ade_collate,
        persistent_workers=False  # Disabled - causes hangs in multi-GPU training
    )
else:
    downstream_train_loader = DataLoader(
        ade_train_dataset,
        batch_size=batch_size_downstream,
        shuffle=True,
        num_workers=2,  # Reduced from 4
        pin_memory=True,
        collate_fn=ade_collate,
        persistent_workers=False  # Disabled - causes hangs
    )
    downstream_val_loader = DataLoader(
        ade_val_dataset,
        batch_size=batch_size_downstream,
        shuffle=False,
        num_workers=2,  # Reduced from 4
        pin_memory=True,
        collate_fn=ade_collate,
        persistent_workers=False  # Disabled - causes hangs
    )

if is_main_process:
    print(f"Train dataset: {len(ade_train_dataset):,} samples")
    print(f"Val dataset: {len(ade_val_dataset):,} samples")
    print(f"Train DataLoader: {len(downstream_train_loader)} batches per GPU")
    print(f"Val DataLoader: {len(downstream_val_loader)} batches per GPU")
    if world_size > 1:
        print(f"Using DistributedSampler for {world_size} GPUs")

IGNORE_INDEX = 255
NUM_CLASSES  = 150

# -----------------------
# JEPA Segmentation Model (uses SegmentationHead from model.py)
# -----------------------

class JEPASegmentationModel(nn.Module):
    """
    JEPA backbone + pixel decoder -> Enhanced SegmentationHead.
    """
    def __init__(self, backbone_model, num_classes=150, mid_channels=128):
        super().__init__()
        self.backbone = backbone_model.context_encoder
        self.pixel_decoder = backbone_model.pixel_decoder
        self.embed_dim = backbone_model.embed_dim
        self.ds16 = backbone_model.ds16
        self.ds32 = backbone_model.ds32
        self.head = SegmentationHead(self.embed_dim, mid_channels, num_classes, dropout=0.4)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens, (enc_h, enc_w) = self.backbone(x)                 # [B, P, D]
        feat = tokens.transpose(1,2).reshape(B, self.embed_dim, enc_h, enc_w)

        # pyramid as in pretrain
        C3  = F.interpolate(feat, size=(H//8,  W//8),  mode='bilinear', align_corners=False)
        x16 = self.ds16(C3)
        x32 = self.ds32(x16)
        C4  = F.interpolate(x16, size=(H//16, W//16), mode='bilinear', align_corners=False)
        C5  = F.interpolate(x32, size=(H//32, W//32), mode='bilinear', align_corners=False)

        Fi1, F_last = self.pixel_decoder([C3, C4, C5], (H, W))    # Fi1 ~ s/8, F_last ~ s/4
        return self.head(Fi1, F_last, (H, W))                     # [B, K, H, W]

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
    plt.suptitle(f"Epoch {epoch} - Segmentation Results", fontsize=16)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

# -----------------------
# One-time ADE sanity peek
# -----------------------
if is_main_process:
    try:
        peek = next(iter(downstream_train_loader))
        print("ADE20K uniques (peek):", torch.unique(peek["masks"])[:20].cpu())
    except Exception as e:
        print("ADE peek skipped:", e)

# -----------------------
# Load JEPA parts & build model
# -----------------------
if is_main_process:
    print("Loading pretrained JEPA model...")
    
jepa_model = MaskJEPA2D(
    in_chans=3, tau=0.996, fi1_mask_ratio=0.5,
    num_queries=50, num_cross_attn=5, num_self_attn=1, patch_size=8
).to(device)

weights_path = "/u/ssood/projects/Rl-JEPA/jepa_rl_training_output_QUICK_TEST_A2C_128/mask_jepa_rl_pretrained_weights.pt"
if not os.path.exists(weights_path):
    if is_main_process:
        print(f"ERROR: Pretrained JEPA weights not found at {weights_path}")
        print("Please run train.py first to pretrain the JEPA model!")
    exit(1)

# Load pretrained weights
ckpt = torch.load(weights_path, map_location=device)
jepa_model.context_encoder.load_state_dict(ckpt['backbone_state_dict'])
jepa_model.pixel_decoder.load_state_dict(ckpt['pixel_decoder_state_dict'])
if 'transformer_decoder_cross_blocks_state_dict' in ckpt:
    # Load cross-attention blocks if available
    pass  # These are for predictor, not needed for fine-tuning
if is_main_process:
    print("Loaded: context_encoder, pixel_decoder from pretrained weights")

model = JEPASegmentationModel(jepa_model, num_classes=NUM_CLASSES).to(device)  # Uses 128 mid_channels by default
model = model.to(memory_format=torch.channels_last)

# Fix BatchNorm for multi-GPU training
if world_size > 1:
    # Convert BatchNorm to SyncBatchNorm for proper multi-GPU training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if is_main_process:
        print("Converted BatchNorm to SyncBatchNorm for multi-GPU training")
    
    # Wrap with DDP - FIXED DEVICE IDS
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if is_main_process:
        print(f"Model wrapped with DDP across {world_size} GPUs")
else:
    if is_main_process:
        print("Using single GPU training")

# -----------------------
# Training Configuration - MIMICKING SUPERVISED.PY
# -----------------------
num_epochs = 100 # full training is 300
print_every = 100

# Split learning rates for backbone vs head (MATCHING SUPERVISED.PY)
backbone_lr = 5e-4  # MATCHING supervised.py
head_lr = 5e-3      # MATCHING supervised.py     
weight_decay = 0.01
warmup_epochs = 0   # MATCHING supervised.py

# Early stopping config (MATCHING SUPERVISED.PY)
early_stop_patience = 20
best_val_miou = 0.0
epochs_no_improve = 0

# Split parameters for different learning rates
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'backbone' in name or 'encoder' in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

# Create optimizer with different learning rates
optimizer = AdamW([
    {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
    {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
])

# Learning rate scheduler with warmup (MATCHING SUPERVISED.PY)
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: lr_lambda(epoch, num_epochs, warmup_epochs, min_lr_ratio=0.01)
)

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
scaler = GradScaler('cuda')

save_dir = "./jepa_finetuning_output_QUICK_TEST_A2C_128"
# Ensure directory exists on all ranks
os.makedirs(save_dir, exist_ok=True)
if is_main_process:
    print(f"Created save directory: {save_dir}")

# CSV logging setup (rank 0 only)
csv_path = os.path.join(save_dir, "finetuning_log.csv")
if is_main_process:
    # Ensure directory exists before writing CSV
    os.makedirs(save_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'epoch_time_min', 'train_ce', 'train_miou', 'val_ce', 'val_miou', 'lr'])

if is_main_process:
    print("Starting JEPA fine-tuning (mimicking supervised.py settings)...")
    print(f"Train batches: {len(downstream_train_loader)}, Val batches: {len(downstream_val_loader)}")
    print(f"Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

train_ce_hist, val_ce_hist, val_miou_hist = [], [], []

# NCCL environment variables already set at top of file

# Performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Better performance

# Access base model for saving
base_model = model.module if hasattr(model, 'module') else model

# Track training time
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    # Set epoch for DistributedSampler to ensure proper shuffling
    if world_size > 1:
        downstream_train_loader.sampler.set_epoch(epoch)
        
    model.train()
    epoch_ce = 0.0
    epoch_meter = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)
    print_meter = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)

    for batch_idx, batch in enumerate(downstream_train_loader):
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
            print(f"  Batch {batch_idx:4d}/{len(downstream_train_loader)} | CE: {loss.item():.4f} | mIoU: {miou_b:.4f}")
            print_meter.reset()

        del images, masks, logits, loss

    epoch_ce /= len(downstream_train_loader)
    train_miou = epoch_meter.get()  # Get training metrics
    train_ce_hist.append(epoch_ce)
    scheduler.step()

    # Validation every epoch
    model.eval()
    val_ce = 0.0
    val_meter = StreamingSegMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)
    
    with torch.no_grad():
        for batch in downstream_val_loader:
            images = batch["images"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks  = batch["masks"].to(device, non_blocking=True)
            with autocast('cuda', enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(images)
                ce = criterion(logits, masks)
            val_ce += ce.item()
            val_meter.update(logits, masks)
            del images, masks, logits, ce

    val_ce /= len(downstream_val_loader)
    val_miou = val_meter.get()
    val_ce_hist.append(val_ce)
    val_miou_hist.append(val_miou)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    if is_main_process:
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train CE: {epoch_ce:.4f} mIoU: {train_miou:.4f} | Val CE: {val_ce:.4f} mIoU: {val_miou:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Log to CSV
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{epoch_duration/60:.2f}", f"{epoch_ce:.4f}", f"{train_miou:.4f}",
                           f"{val_ce:.4f}", f"{val_miou:.4f}", f"{scheduler.get_last_lr()[0]:.2e}"])

    # Early stopping and save best by validation mIoU
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        epochs_no_improve = 0
        if is_main_process:
            best_path = os.path.join(save_dir, "best_jepa_model.pt")
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
                print(f"  [Early Stop] Patience exceeded. Stopping fine-tuning.")
            break

    # Synchronize before visualization
    if world_size > 1:
        dist.barrier()
    
    # Visualizations every 20 epochs (rank 0 only)
    if (epoch + 1) % 20 == 0 and is_main_process:
        with torch.no_grad():
            vis_batch = next(iter(downstream_val_loader))
            vis_images = vis_batch["images"].to(device).to(memory_format=torch.channels_last)
            vis_masks  = vis_batch["masks"].to(device)
            with autocast('cuda', enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                vis_logits = model(vis_images)
            vis_path = os.path.join(save_dir, f"jepa_epoch_{epoch+1:03d}.png")
            visualize_segmentation(vis_images, vis_masks, vis_logits, epoch+1, vis_path)
            print(f"  Saved visualization: {vis_path}")
            del vis_batch, vis_images, vis_masks, vis_logits
    
    # Synchronize all ranks after visualization
    if world_size > 1:
        dist.barrier()

    torch.cuda.empty_cache(); gc.collect()

if is_main_process:
    total_training_time = time.time() - training_start_time
    print("JEPA fine-tuning completed!")
    print(f"Best Val mIoU: {best_val_miou:.4f}")
    print(f"Total JEPA fine-tuning time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    
    # Log total time to CSV
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TOTAL', f"{total_training_time/60:.1f}", '', '', '', '', ''])

# Save final (rank 0 only)
if is_main_process:
    final_path = os.path.join(save_dir, "final_jepa_model.pt")
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'train_ce_losses': train_ce_hist,
        'val_ce_losses': val_ce_hist,
        'val_mious': val_miou_hist,
        'best_val_miou': best_val_miou
    }, final_path)
    print(f"Final JEPA model saved: {final_path}")

# Cleanup DDP
cleanup_ddp()
