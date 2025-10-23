#!/usr/bin/env python3
"""
Multi-GPU DDP Training Script for MaskJEPA

Usage:
    # Single GPU:
    python train.py
    
    # Multi-GPU (automatic detection):
    torchrun --nproc_per_node=auto train.py
    
    # Specific number of GPUs:
    torchrun --nproc_per_node=4 train.py
"""

# Set NCCL environment variables before importing torch
import os
os.environ['NCCL_TIMEOUT'] = '7200'              # 2 hours
os.environ['NCCL_BLOCKING_WAIT'] = '1'           # Synchronous error handling
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'    # Better error reporting
os.environ['NCCL_IB_DISABLE'] = '1'              # Disable InfiniBand
os.environ['NCCL_SOCKET_NTHREADS'] = '4'         # Reduce overhead
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'        # Reduce overhead

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import matplotlib.pyplot as plt
import numpy as np
import gc
import math
import time   # for epoch timing
import atexit
import csv

from MaskJEPA import MaskJEPA2D
from utils import visualize_jepa_patch_quality, lr_lambda
from Dataloader import batch_size_pretrain
from Dataset import JEPADataset, jepa_collate
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from rl_agent import MaskingAgentTrainer, calculate_semantic_coherence, calculate_jepa_rewards


use_bf16 = torch.cuda.is_bf16_supported()


USE_RL_MASKING = True
QUICK_TEST = False


# DDP Setup
def setup_ddp():
    if all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']):
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        return rank, world_size, device
    else:
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to single CPU")
            return 0, 1, "cpu"
        
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s): {[f'GPU {i}' for i in range(gpu_count)]}")
        
        if gpu_count > 1:
            print(f"You have {gpu_count} GPUs! For multi-GPU training, run:")
            print(f"   torchrun --nproc_per_node={gpu_count} train.py")
            print("Falling back to single GPU training for now...")
        
        return 0, 1, "cuda:0"

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

rank, world_size, device = setup_ddp()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main_process = rank == 0

atexit.register(cleanup_ddp)
1
jepa_dataset = JEPADataset()

if QUICK_TEST:
    jepa_dataset = Subset(jepa_dataset, range(min(1000, len(jepa_dataset))))
    if is_main_process:
        print(f"QUICK TEST MODE: Limited dataset to {len(jepa_dataset)} samples")
if world_size > 1:
    train_sampler = DistributedSampler(jepa_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    pretrain_loader = DataLoader(
        jepa_dataset,
        batch_size=batch_size_pretrain,
        sampler=train_sampler,
        num_workers=6,
        pin_memory=True,
        collate_fn=jepa_collate
    )
else:
    pretrain_loader = DataLoader(
        jepa_dataset,
        batch_size=batch_size_pretrain,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=jepa_collate
    )

if is_main_process:
    print(f"Dataset size: {len(jepa_dataset):,} samples")
    print(f"DataLoader: {len(pretrain_loader)} batches per GPU")
    if world_size > 1:
        print(f"Using DistributedSampler for {world_size} GPUs")

num_epochs =  10 
warmup_epochs = 2
base_lr = 1e-4
weight_decay = 0.05

early_stop_patience = 15
best_total_sc = float('inf')
epochs_no_improve = 0

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def enable_gradient_checkpointing(model):
    if hasattr(model.context_encoder, 'vit') and hasattr(model.context_encoder.vit, 'blocks'):
        for block in model.context_encoder.vit.blocks:
            if hasattr(block, 'set_grad_checkpointing'):
                block.set_grad_checkpointing(True)
    if hasattr(model.target_encoder, 'vit') and hasattr(model.target_encoder.vit, 'blocks'):
        for block in model.target_encoder.vit.blocks:
            if hasattr(block, 'set_grad_checkpointing'):
                block.set_grad_checkpointing(True)

if is_main_process:
    print("Creating model...")
model = MaskJEPA2D(
    in_chans=3,
    tau=0.996,
    fi1_mask_ratio=0.5,
    num_queries=32,
    num_cross_attn=9,
    num_self_attn=2,
    patch_size=8
).to(device)

D = model.embed_dim

enable_gradient_checkpointing(model)

if world_size > 1:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if is_main_process:
        print("Converted BatchNorm to SyncBatchNorm for multi-GPU training")
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if is_main_process:
        print(f"Model wrapped with DDP across {world_size} GPUs")
else:
    if is_main_process:
        print("Using single GPU training")

if is_main_process:
    print("Gradient checkpointing enabled")

scaler = GradScaler('cuda')

optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, num_epochs, warmup_epochs))

if USE_RL_MASKING:
    save_dir = "./jepa_rl_training_output"
    model_filename = "mask_jepa_rl_pretrained_weights.pt"
else:
    save_dir = "./jepa_training_output"
    model_filename = "mask_jepa_pretrained_weights.pt"

os.makedirs(save_dir, exist_ok=True)
best_ckpt_path = os.path.join(save_dir, model_filename)

if is_main_process:
    print(f"Training mode: {'RL-Masking' if USE_RL_MASKING else 'Standard'}")
    print(f"Save directory: {save_dir}")
    print(f"Model filename: {model_filename}")

csv_path = os.path.join(save_dir, "training_log.csv")
if is_main_process:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'epoch_time_min', 'total_sc', 'recon_sc', 'denoise_sc', 'lr', 'gpu_memory_gb'])

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

if is_main_process:
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Batch size: {batch_size_pretrain}")

base_model = model.module if hasattr(model, 'module') else model
ps = base_model.patch_size
mr = base_model.fi1_mask_ratio
nq = base_model.predictor.num_queries
nca = len(base_model.predictor.cross_blocks)
nsa = len(base_model.predictor.self_blocks)

if is_main_process:
    print(f"Model config: patch_size={ps}, mask_ratio={mr}, queries={nq}, cross_attn={nca}, self_attn={nsa}")

planned_updates_per_epoch = len(pretrain_loader)
max_updates = num_epochs * planned_updates_per_epoch
global_update = 0

rl_trainer = None
if USE_RL_MASKING:
    fi1_shape = (1, 96, 64, 64)
    rl_trainer = MaskingAgentTrainer(fi1_shape, mask_ratio=0.5, patch_size=8, device=device)
    if is_main_process:
        print(f"RL masking enabled - trainer initialized on CPU with fi1_shape: {fi1_shape}")
else:
    if is_main_process:
        print("Using random masking (RL disabled)")

train_losses = []
best_snapshot = None
training_start_time = time.time()

for epoch in range(num_epochs):
    if world_size > 1:
        pretrain_loader.sampler.set_epoch(epoch)
        
    epoch_start_time = time.time()
    if is_main_process:
        print(f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_denoise_loss = 0.0
    
    clear_memory()
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pretrain_loader):
        images = batch["images"].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16):
            
            if USE_RL_MASKING and rl_trainer:

                rl_start = time.time()
                
                actual_batch_size = images.shape[0]
                rl_sub_batch_size = 8
                all_rl_masks = []
                all_episodes = []
                
                for sub_start in range(0, actual_batch_size, rl_sub_batch_size):
                    sub_end = min(sub_start + rl_sub_batch_size, actual_batch_size)
                    sub_size = sub_end - sub_start
                    
                    masks, eps = rl_trainer.generate_masks_for_batch(batch_size=sub_size)
                    all_rl_masks.extend(masks)
                    all_episodes.extend(eps)
                
                batched_masks = torch.stack(all_rl_masks, dim=0).to(device)
                
                rl_time = time.time() - rl_start
                if batch_idx % 10 == 0 and is_main_process:
                    avg_episode_length = np.mean([len(ep['actions']) for ep in all_episodes])
                    print(f"RL mask gen time: {rl_time:.2f}s, avg episode length: {avg_episode_length:.1f}")
                
                outputs = model(images, external_fi1_mask=batched_masks)
                episodes = all_episodes
            else:
                outputs = model(images)
            
            pred = outputs['predicted_features']
            tgt = outputs['target_masked']
            idx = outputs['mask_indices']
            valid = (idx >= 0).unsqueeze(-1)
            
            # Reconstruction loss
            if pred.numel() == 0 or valid.sum() == 0:
                recon_loss = pred.new_zeros(())
            else:
                diff = (pred - tgt) * valid
                recon_loss = diff.pow(2).sum() / valid.sum().clamp_min(1)

            # Option 2: Predict clean image x (current default)
            x4 = F.interpolate(
                outputs['original_input'],
                size=outputs['denoised_prediction'].shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            denoise_loss = F.mse_loss(outputs['denoised_prediction'], x4)
            
            if batch_idx == 0 and is_main_process:
                td = F.interpolate(images, size=outputs['denoised_prediction'].shape[-2:],
                                   mode='bilinear', align_corners=False)
                pd = outputs['denoised_prediction'].detach()
                print(f"[probe] target: mean={td.mean().item():.3f} std={td.std().item():.3f}")
                print(f"[probe] pred  : mean={pd.mean().item():.3f} std={pd.std().item():.3f}")

            total_loss = recon_loss + denoise_loss
        
        if USE_RL_MASKING and rl_trainer is not None:
            pixel_errors_batch = []
            
            for b in range(outputs['predicted_features'].shape[0]):
                pred_b = outputs['predicted_features'][b].detach().cpu().float().numpy()
                tgt_b = outputs['target_masked'][b].detach().cpu().float().numpy()
                
                if pred_b.size == 0:
                    patch_errors = np.zeros((0,), dtype=np.float32)
                else:
                    diff = pred_b - tgt_b
                    patch_errors = (diff * diff).mean(axis=-1)
                
                pixel_errors_batch.append(patch_errors)
            
            jepa_outputs_for_rl = {
                'pixel_errors': pixel_errors_batch,
                'features': outputs['fi1_features'],
                'mask_indices': outputs['mask_indices']
            }
            
            real_rewards = calculate_jepa_rewards(episodes, jepa_outputs_for_rl)
            
        # UPDATE_FREQUENCY = 20
        # if batch_idx % UPDATE_FREQUENCY == 0 and is_main_process:
        #     print(f"Updating RL agent at batch {batch_idx}/{len(pretrain_loader)}")
        #     update_start = time.time()
        #     rl_trainer.update_agent(episodes, jepa_outputs_for_rl)
        #     update_time = time.time() - update_start
        #     print(f"RL update time: {update_time:.2f}s")

        if is_main_process:
            print(f"Updating RL agent at batch {batch_idx}/{len(pretrain_loader)}")
            update_start = time.time()
            rl_trainer.update_agent(episodes, jepa_outputs_for_rl)
            update_time = time.time() - update_start
            print(f"RL update time: {update_time:.2f}s")
        
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        progress = global_update / max(1, max_updates - 1)
        tau_now = base_model.tau_base + (base_model.tau_final - base_model.tau_base) * progress
        base_model.set_ema_tau(tau_now)
        base_model.update_ema()
        global_update += 1
        
        optimizer.zero_grad()
        
        epoch_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_denoise_loss += denoise_loss.item()
        
        del outputs, recon_loss, denoise_loss, total_loss, images
        
        if batch_idx % 50 == 0 and is_main_process:
            recon_sc = (epoch_recon_loss / (batch_idx + 1)) / D
            denoise_sc = (epoch_denoise_loss / (batch_idx + 1))
            total_sc = recon_sc + denoise_sc
            
            print(f"  Batch {batch_idx}/{len(pretrain_loader)} - "
                  f"TotalSc: {total_sc:.4f}, ReconSc: {recon_sc:.4f}, DenoiseSc: {denoise_sc:.4f}")
    
    epoch_loss /= len(pretrain_loader)
    epoch_recon_loss /= len(pretrain_loader)
    epoch_denoise_loss /= len(pretrain_loader)
    
    train_losses.append(epoch_loss)
    scheduler.step()
    
    recon_sc_epoch = epoch_recon_loss / D
    denoise_sc_epoch = epoch_denoise_loss
    total_sc_epoch = recon_sc_epoch + denoise_sc_epoch
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    if is_main_process:
        print(f"  Avg losses - TotalSc: {total_sc_epoch:.4f}, "
              f"ReconSc: {recon_sc_epoch:.4f}, DenoiseSc: {denoise_sc_epoch:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"  Time for epoch {epoch+1}: {epoch_duration/60:.2f} minutes")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{epoch_duration/60:.2f}", f"{total_sc_epoch:.4f}", 
                           f"{recon_sc_epoch:.4f}", f"{denoise_sc_epoch:.4f}", 
                           f"{scheduler.get_last_lr()[0]:.2e}", f"{torch.cuda.memory_allocated()/1e9:.2f}"])
    
    if (epoch + 1) % 1 == 0 and is_main_process:
        print("  Running evaluation...")
        model.eval()
        
        with torch.no_grad():
            eval_batch = next(iter(pretrain_loader))
            eval_images = eval_batch["images"][:4].to(device)
            
            with autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16):
                eval_outputs = model(eval_images)
            
            H, W = eval_images.shape[-2:]
            fi1_tile = max(H // (H // 8), 1)
            
            if USE_RL_MASKING:
                vis_filename = f"jepa_rl_epoch_{epoch+1:03d}.png"
            else:
                vis_filename = f"jepa_epoch_{epoch+1:03d}.png"
            vis_path = os.path.join(save_dir, vis_filename)
            
            visualize_jepa_patch_quality(
                original=eval_images,
                predicted_features=eval_outputs['predicted_features'].float(),
                target_features=eval_outputs['target_masked'].float(),
                patch_mask=eval_outputs['fi1_mask'],
                epoch=epoch+1,
                save_path=vis_path,
                patch_size=fi1_tile
            )
            
            del eval_batch, eval_images, eval_outputs
        
        print(f"    Saved visualization: {vis_path}")
        
    model.train()
    clear_memory()
    
    if total_sc_epoch < best_total_sc:
        best_total_sc = total_sc_epoch
        epochs_no_improve = 0
        if is_main_process:
            best_snapshot = {
                'backbone_state_dict': base_model.context_encoder.state_dict(),
                'pixel_decoder_state_dict': base_model.pixel_decoder.state_dict(),
                'transformer_decoder_cross_blocks_state_dict': base_model.predictor.cross_blocks.state_dict(),
            }
            torch.save(best_snapshot, best_ckpt_path)
            print(f"    [best] New best TotalSc={best_total_sc:.4f}. Saved: {best_ckpt_path}")
    else:
        epochs_no_improve += 1
        if is_main_process:
            print(f"  [early-stop] No improvement ({epochs_no_improve}/{early_stop_patience}).")
        if epochs_no_improve >= early_stop_patience:
            if is_main_process:
                print(f"  [early-stop] Patience exceeded. Stopping training early.")
            break

if is_main_process:
    total_training_time = time.time() - training_start_time
    print("Training completed!")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TOTAL', f"{total_training_time/60:.1f}", '', '', '', '', ''])

if is_main_process:
    print(f"Training completed. Best model already saved: {best_ckpt_path}")

cleanup_ddp()
