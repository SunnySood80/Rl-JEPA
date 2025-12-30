import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

# Global seed variable (set by set_seed())
_GLOBAL_SEED = None

def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    Call this FIRST before importing models or creating any stochastic operations.
    
    Seeds:
    - Python random
    - NumPy random
    - PyTorch random (CPU)
    - PyTorch random (CUDA - all devices)
    - PyTorch deterministic mode (slower but fully reproducible)
    - Python hash randomization (for dict/set ordering)
    
    Args:
        seed: Integer seed value
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Seed all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic mode (slower but fully reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed for reproducible dict/set ordering
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Seed set to {seed} (Python, NumPy, PyTorch CPU/CUDA, deterministic mode)")

def get_seed() -> int:
    """Get the current global seed."""
    if _GLOBAL_SEED is None:
        raise RuntimeError("Seed not set! Call set_seed() first before any stochastic operations.")
    return _GLOBAL_SEED


def generate_fi1_mask(fi1_shape: tuple, mask_ratio: float = 0.5, patch_size: int = 8, device='cuda'):
    B, D, H8, W8 = fi1_shape  # e.g., [B, D, 28, 28] for 224x224 images
    
    # Calculate number of patches
    n_patches_h = H8 // patch_size  # 28/8 = 3
    n_patches_w = W8 // patch_size  # 28/8 = 3
    total_patches = n_patches_h * n_patches_w  # 9 patches total
    
    num_masked = int(mask_ratio * total_patches)  # e.g., 4 patches masked
    
    # Generate mask
    fi1_mask = torch.zeros(B, H8 * W8, dtype=torch.bool, device=device)
    
    seed = get_seed()
    for b in range(B):
        # Create generator with seed for reproducible masking (different seed per batch for variety)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + b)  # Offset by batch index for different masks per batch
        # Randomly select which patches to mask (seeded for reproducibility)
        masked_patch_ids = torch.randperm(total_patches, device=device, generator=generator)[:num_masked]
        
        for patch_id in masked_patch_ids:
            # Convert patch_id to patch coordinates
            ph = patch_id // n_patches_w
            pw = patch_id % n_patches_w
            
            # Convert to pixel coordinates in Fi1
            h_start = ph * patch_size
            h_end = min(h_start + patch_size, H8)
            w_start = pw * patch_size  
            w_end = min(w_start + patch_size, W8)
            
            # Mask this patch in flattened Fi1
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    fi1_mask[b, h * W8 + w] = True
    
    return fi1_mask  # [B, H8*W8]



def compute_patch_grid(image_shape, patch_size):
    """
    image_shape expected: (C, H, W)
    """
    _, H, W = image_shape
    n_h = H // patch_size
    n_w = W // patch_size
    P = n_h * n_w
    cropped_shape = (n_h * patch_size, n_w * patch_size)
    return n_h, n_w, P, cropped_shape


def extract_patch_embeddings_from_feature_map(feats: torch.Tensor) -> torch.Tensor:
    """
    feats: [N, D, n_h, n_w]  -> returns [N, P, D]  (P = n_h * n_w)
    """
    N, D, n_h, n_w = feats.shape
    # Move D to last, then flatten spatial dims
    return feats.permute(0, 2, 3, 1).reshape(N, -1, D)


def compute_denoising_loss(self, denoised_prediction, original_input):
    # Downsample target
    target_downsampled = F.interpolate(
        original_input, 
        size=denoised_prediction.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    return F.mse_loss(denoised_prediction, target_downsampled)

def compute_reconstruction_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute reconstruction loss for predicted vs target embeddings
    """
    return F.mse_loss(preds, targets, reduction="mean")

@torch.no_grad()
def update_ema(target_net: nn.Module, online_net: nn.Module, tau: float):
    """
    Update target network parameters using exponential moving average
    """
    for t_param, s_param in zip(target_net.parameters(), online_net.parameters()):
        t_param.data.mul_(tau).add_(s_param.data, alpha=1 - tau)

def unpatchify_embeddings(emb: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
    """
    Convert patch embeddings back to 2D feature map
    emb: [N, P, D] -> [N, D, n_h, n_w]
    """
    N, P, D = emb.shape
    emb_4d = emb.view(N, n_h, n_w, D)
    return emb_4d.permute(0, 3, 1, 2).contiguous()


def apply_fi1_mask_tokens(fi1_features: torch.Tensor, fi1_mask: torch.Tensor, mask_token: torch.Tensor):
    """
    Apply masking to Fi1 features using learned mask tokens
    
    Args:
        fi1_features: (B, D, H8, W8) Fi1 feature maps
        fi1_mask: (B, H8*W8) boolean mask
        mask_token: (1, D, 1, 1) learned mask token
    
    Returns:
        masked_fi1: Fi1 with mask tokens at masked positions
    """
    B, D, H8, W8 = fi1_features.shape
    
    # Reshape mask
    mask_2d = fi1_mask.reshape(B, H8, W8).unsqueeze(1).expand(-1, D, -1, -1)
    
    # Replace masked positions
    masked_fi1 = torch.where(mask_2d, mask_token.expand(B, D, H8, W8), fi1_features)
    
    return masked_fi1

def visualize_jepa_patch_quality(original: torch.Tensor, predicted_features: torch.Tensor, target_features: torch.tensor, patch_mask: torch.Tensor, patch_size: int,
                                 epoch: int, save_path: str):
    
    print(f"Mask shape: {patch_mask.shape}")
    print(f"Total True across all 4 images: {patch_mask.sum().item()}")
    print(f"First image True count: {patch_mask[0].sum().item()}")

    original0 = original[0].cpu().numpy()
    original0 = original0.transpose(1, 2, 0)

    def _to_display_img(x: torch.Tensor) -> np.ndarray:
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            xc = x.clone()
            mn, mx = float(xc.min()), float(xc.max())
            if 0.0 <= mn and mx <= 1.0:
                pass  # already [0,1]
            elif -3.5 <= mn <= 3.5 and -3.5 <= mx <= 3.5:
                # assume ImageNet norm
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                xc = xc * std + mean
            else:
                # min-max to [0,1]
                xc = (xc - mn) / (max(mx - mn, 1e-6))
            img = xc.permute(1, 2, 0).numpy()
        else:
            arr = x.numpy()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            img = arr
        return np.clip(img, 0.0, 1.0)
    
    # Use it
    original0 = _to_display_img(original[0])
    

    mask0 = patch_mask[0]
    pixel_indices = mask0.nonzero(as_tuple=False).squeeze(1).cpu().numpy()

    targ0 = target_features[0].detach().cpu().numpy()
    pred0 = predicted_features[0].detach().cpu().numpy()
    diff = targ0 - pred0
    diff_squared = diff * diff
    per_patch_loss = diff_squared.mean(axis=-1)
    lo, hi = float(per_patch_loss.min()), float(per_patch_loss.max())
    denom = hi - lo

    normalized_patch_loss = (1.0 - (per_patch_loss - lo) / denom)

    feature_map_size = int(np.sqrt(len(mask0)))
    patches_per_row = feature_map_size // patch_size
    

    patch_errors = {}

    for i, pixel_idx in enumerate(pixel_indices):

        pixel_row = pixel_idx // feature_map_size
        pixel_col = pixel_idx % feature_map_size

        patch_row = pixel_row // patch_size
        patch_col = pixel_col // patch_size

        patch_idx = patch_row * patches_per_row + patch_col
        error_val = normalized_patch_loss[i]

        if patch_idx not in patch_errors:
            patch_errors[patch_idx] = []

        # each key is a unique patch idx, each patch idx correpsonds to pixel errors
        patch_errors[patch_idx].append(error_val)

    # grab all the unique patch idxs, average the rrors for all of them
    unique_patches = sorted(list(patch_errors.keys()))
    averaged_errors = [np.mean(patch_errors[p]) for p in unique_patches]

    scale = original0.shape[0] / feature_map_size
    colormap = plt.get_cmap('RdYlGn')
    masked_img0 = original0.copy()

    print(f"feature map size: {feature_map_size}")
    print(f"original image size: {original0.shape[0]}")
    print(f"scale: {scale}")
    print(f"patch size: {patch_size}")

    counter = 0
    for i, patch_idx in enumerate(unique_patches):
        patch_row = patch_idx // patches_per_row
        patch_col = patch_idx % patches_per_row

        feature_h_start = patch_row * patch_size
        feature_w_start = patch_col * patch_size

        original_h_start = int(feature_h_start * scale)
        original_w_start = int(feature_w_start * scale)
        original_patch_size = int(patch_size * scale)

        error_to_color = averaged_errors[i]
        color = colormap(error_to_color)[:3]
        
        counter += 1
        masked_img0[original_h_start:original_h_start+original_patch_size, 
               original_w_start:original_w_start+original_patch_size] = color
    print(f'here is the amount of patches in this pic: {counter}')


    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    axs[0].imshow(original0, interpolation='nearest')
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    axs[1].imshow(masked_img0, interpolation='nearest')
    axs[1].set_title('Masked image with patch errors')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# def visualize_jepa_patch_quality(
#     original: torch.Tensor,
#     predicted_features: torch.Tensor,
#     target_features: torch.Tensor,
#     patch_mask: torch.Tensor,
#     epoch: int,
#     save_path: str,
#     patch_size: int,
# ):

#     print(f"VIZ INPUT - patch_mask shape: {patch_mask.shape}")
#     print(f"VIZ INPUT - Total True in mask: {patch_mask.sum().item()}")
#     print(f"VIZ INPUT - Total False in mask: {(~patch_mask).sum().item()}")

#     # ----- robust image-to-display -----
#     def _to_display_img(x: torch.Tensor) -> np.ndarray:
#         x = x.detach().cpu()
#         if x.ndim == 3 and x.shape[0] in (1, 3):
#             xc = x.clone()
#             mn, mx = float(xc.min()), float(xc.max())
#             if 0.0 <= mn and mx <= 1.0:
#                 pass  # already [0,1]
#             elif -3.5 <= mn <= 3.5 and -3.5 <= mx <= 3.5:
#                 # assume ImageNet norm
#                 mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#                 std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#                 xc = xc * std + mean
#             else:
#                 # min-max to [0,1]
#                 xc = (xc - mn) / (max(mx - mn, 1e-6))
#             img = xc.permute(1, 2, 0).numpy()
#         else:
#             arr = x.numpy()
#             arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
#             img = arr
#         return np.clip(img, 0.0, 1.0)

#     # ---- prep first image ----
#     original_np = _to_display_img(original[0])
#     H, W = original_np.shape[:2]
#     n_h, n_w = H // patch_size, W // patch_size

#     # ---- per-masked-tile losses ----
#     pred0 = predicted_features[0].detach().cpu().numpy()   # [M, D]
#     tgt0  = target_features[0].detach().cpu().numpy()      # [M, D]
#     if pred0.size == 0:
#         per_patch_losses = np.zeros((0,), dtype=np.float32)
#     else:
#         diff = pred0 - tgt0
#         per_patch_losses = (diff * diff).mean(axis=-1)     # [M]

#     if per_patch_losses.size > 0:
#         lo, hi = float(per_patch_losses.min()), float(per_patch_losses.max())
#         denom = (hi - lo) if (hi > lo) else 1.0
#         normalized_quality = 1.0 - ((per_patch_losses - lo) / denom)
#     else:
#         normalized_quality = np.zeros((0,), dtype=np.float32)

#     # ---- masked indices ----
#     mask0 = patch_mask[0].detach().cpu().view(-1)          # [P]
#     masked_indices = mask0.nonzero(as_tuple=False).squeeze(1).numpy()  # [K]

#     feature_map_size = int(np.sqrt(len(mask0)))
#     map_masked_pixels = len(masked_indices)


#     # ---- figure ----
#     fig, axs = plt.subplots(1, 3, figsize=(14, 5))

#     # Left: original
#     axs[0].imshow(original_np, interpolation='nearest')
#     axs[0].set_title("Original Image")
#     axs[0].axis('off')

#     # Center: mask overlay (red) with black grid
#     masked_img = original_np.copy()
#     overlay = masked_img.copy()
#     red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

#     for pidx in masked_indices:
#         if pidx < 0 or pidx >= n_h * n_w:
#             continue
#         ih, iw = divmod(int(pidx), n_w)
#         h0, h1 = ih * patch_size, (ih + 1) * patch_size
#         w0, w1 = iw * patch_size, (iw + 1) * patch_size
#         overlay[h0:h1, w0:w1, :] = red

#     alpha_center = 0.35
#     masked_img = (1 - alpha_center) * masked_img + alpha_center * overlay
#     axs[1].imshow(np.clip(masked_img, 0.0, 1.0), interpolation='nearest')
#     axs[1].set_title(f"Epoch {epoch} - JEPA Analysis\nMasked Patches (Red)\n{int(mask0.sum())}/{mask0.numel()} masked")
#     axs[1].axis('off')

#     # Right: reconstruction quality (bold colored squares)
#     quality_img = original_np.copy()
#     colormap = plt.get_cmap('RdYlGn')  # green=good, red=bad
#     alpha_patch = 0.85                 # strong overlay for bold squares
#     grid_thick = max(1, patch_size // 16)  # thicker grid lines

#     limit = min(len(normalized_quality), len(masked_indices))
#     for idx in range(limit):
#         patch_idx = int(masked_indices[idx])
#         if patch_idx < 0 or patch_idx >= n_h * n_w:
#             continue

#         ih, iw = divmod(patch_idx, n_w)
#         h0, h1 = ih * patch_size, (ih + 1) * patch_size
#         w0, w1 = iw * patch_size, (iw + 1) * patch_size

#         q = float(np.asarray(normalized_quality[idx]).mean())
#         if not np.isfinite(q):
#             q = 0.0
#         q = float(np.clip(q, 0.0, 1.0))

#         color = np.asarray(colormap(q))[:3]  # (3,)
#         patch = quality_img[h0:h1, w0:w1, :]
#         quality_img[h0:h1, w0:w1, :] = (1 - alpha_patch) * patch + alpha_patch * color[None, None, :]

#         # thicker black grid lines
#         quality_img[h0:h0+grid_thick, w0:w1, :] = 0.0
#         quality_img[h0:h1, w0:w0+grid_thick, :] = 0.0

#     axs[2].imshow(np.clip(quality_img, 0.0, 1.0), interpolation='nearest')
#     axs[2].set_title("Reconstruction Quality\n(Green=Good, Red=Poor)")
#     axs[2].axis('off')

#     plt.tight_layout()
#     fig.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)



def lr_lambda(epoch, total_epochs, warmup_epochs=0, min_lr_ratio=0.0):
    """
    Unified learning rate scheduler for all training scripts.
    
    Args:
        epoch: Current epoch (0-based)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (default: 0)
        min_lr_ratio: Minimum learning rate as ratio of base_lr (default: 0.0)
    
    Returns:
        Learning rate multiplier
    """
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    t = epoch - warmup_epochs
    T = max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * t / T))  # 1 → 0
    return min_lr_ratio + (1 - min_lr_ratio) * cosine
