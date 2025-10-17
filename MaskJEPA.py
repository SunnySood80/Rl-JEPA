import copy
import math
from typing import Tuple, Optional
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import modules from model.py
from model import ContextEncoder2D, PixelDecoder2D, Predictor2D
from utils import generate_fi1_mask

def update_ema(target, source, tau):
    """Update EMA target with source parameters"""
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(tau).add_(source_param.data, alpha=1 - tau)

class MaskJEPA2D(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_queries: int = 32,
                 num_cross_attn: int = 2,
                 num_self_attn: int = 1,
                 tau: float = 0.996,
                 fi1_mask_ratio: float = 0.5,
                 patch_size: int = 8,
                 model_name: str = "swin_small_patch4_window7_224",
                 pretrained: bool = True):
        super().__init__()

        self.patch_size = patch_size
        
        # === Context encoder (timm SwinV2 with pos_embed) ===
        self.context_encoder = ContextEncoder2D(model_name=model_name, pretrained=pretrained)

        # pull embed_dim & patch_size from the encoder backbone
        self.embed_dim = self.context_encoder.embed_dim
        self.in_chans = in_chans  # needed for denoising head output

        # === Target encoder (EMA, frozen) ===
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # === Pixel decoders ===
        self.pixel_decoder = PixelDecoder2D(
            in_channels=self.embed_dim,
            embed_dim=self.embed_dim
        )
        self.pixel_decoder_ema = copy.deepcopy(self.pixel_decoder)
        for p in self.pixel_decoder_ema.parameters():
            p.requires_grad = False

        # === Downsampling convs for C4 / C5 ===
        self.ds16 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),           nn.GELU(),
        )
        self.ds32 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),           nn.GELU(),
        )

        # === JEPA Predictor ===
        self.predictor = Predictor2D(
            embed_dim=self.embed_dim,
            num_queries=num_queries,
            num_cross_attn=num_cross_attn,
            num_self_attn=num_self_attn
        )

        # === Denoising head ===
        self.denoising_head = nn.Conv2d(self.embed_dim, in_chans, kernel_size=1)

        # === EMA tau config ===
        self.tau = tau
        self.tau_base  = tau
        self.tau_final = 1.0

        self.fi1_mask_ratio = fi1_mask_ratio

    def forward(self, x: torch.Tensor, external_fi1_mask: Optional[torch.Tensor] = None):
        """
        Paper-correct noise path:
          - Add Gaussian noise at s_last=4 and expand
          - Online branch sees noisy input
          - Target branch sees clean input
        """
        B, C, H, W = x.shape
        device = x.device
    
        # ---------- (A) add noise ----------
        s_last = 4
        H4, W4 = H // s_last, W // s_last
        sigma = 0.4
    
        eps_lr = torch.randn(B, C, H4, W4, device=device, dtype=x.dtype) * sigma
        eps_full = eps_lr.repeat_interleave(s_last, dim=2).repeat_interleave(s_last, dim=3)
        x_noisy = x + eps_full
    
        # ---------- (B) ONLINE BRANCH ----------
        tokens_online, (enc_h, enc_w) = self.context_encoder(x_noisy)  # [B,P,D], (Ht,Wt)
        feat_online = tokens_online.transpose(1, 2).reshape(
            B, self.embed_dim, enc_h, enc_w
        )
        
        C3 = F.interpolate(feat_online, size=(H//8, W//8), mode='bilinear', align_corners=False)
        x16 = self.ds16(C3)
        x32 = self.ds32(x16)
        C4  = F.interpolate(x16, size=(H//16, W//16), mode='bilinear', align_corners=False)
        C5  = F.interpolate(x32, size=(H//32, W//32), mode='bilinear', align_corners=False)
        
        f_i1_online, f_last_online = self.pixel_decoder([C3, C4, C5], (H, W))

        # ---------- (C) TARGET BRANCH ----------
        with torch.no_grad():
            tokens_target, _ = self.target_encoder(x)
            feat_target = tokens_target.transpose(1, 2).reshape(
                B, self.embed_dim, enc_h, enc_w
            )
        
            C3t = F.interpolate(feat_target, size=(H//8, W//8), mode='bilinear', align_corners=False)
            x16t = self.ds16(C3t)
            x32t = self.ds32(x16t)
            C4t  = F.interpolate(x16t, size=(H//16, W//16), mode='bilinear', align_corners=False)
            C5t  = F.interpolate(x32t, size=(H//32, W//32), mode='bilinear', align_corners=False)
        
            f_i1_target, _ = self.pixel_decoder_ema([C3t, C4t, C5t], (H, W))

        # ---------- (D) Fi1 MASK ----------
        if external_fi1_mask is not None:
            # Use externally provided mask
            fi1_mask = external_fi1_mask
        else:
            # Default: generate random mask
            fi1_mask = generate_fi1_mask(
                fi1_shape=f_i1_online.shape,
                mask_ratio=self.fi1_mask_ratio,
                patch_size=self.patch_size,
                device=device
            )  # [B, H8*W8] bool
    
        # ---------- (E) PREDICTOR ----------
        predicted_features, masked_indices, q_proj = self.predictor(f_i1_online, fi1_mask)
    
        # ---------- (F) TARGET GATHER + LN ----------
        D = f_i1_target.shape[1]
        fi1_h, fi1_w = f_i1_target.shape[-2:]
        target_seq = f_i1_target.permute(0, 2, 3, 1).reshape(B, fi1_h * fi1_w, D)
        target_seq = F.layer_norm(target_seq, (D,))
    
        if masked_indices.numel() > 0:
            pad_mask = (masked_indices >= 0)
            safe_idx = masked_indices.clamp_min(0)
            b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(safe_idx)
            target_masked_full = target_seq[b_idx, safe_idx]
            target_masked = target_masked_full * pad_mask.unsqueeze(-1).to(target_masked_full.dtype)
        else:
            target_masked = target_seq.new_zeros((B, 0, D))
    
        # ---------- (G) DENOISING ----------
        denoised_prediction = self.denoising_head(f_last_online)
    
        return {
            'predicted_features': predicted_features,
            'target_masked':      target_masked,
            'mask_info':          (q_proj, masked_indices),
            'denoised_prediction': denoised_prediction,
            'original_input':     x,
            'fi1_mask':           fi1_mask,
            'mask_indices':       masked_indices,
            'eps_target':         eps_lr,
            'fi1_features':       f_i1_online
        }

    def set_ema_tau(self, tau: float):
        self.tau = float(tau)

    @torch.no_grad()
    def update_ema(self):
        update_ema(self.target_encoder, self.context_encoder, tau=self.tau)
        update_ema(self.pixel_decoder_ema, self.pixel_decoder, tau=self.tau)
