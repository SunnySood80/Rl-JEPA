import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple, Optional

# --- PatchEmbed2D: remove undefined pos_embed add (or implement it properly) ---
class PatchEmbed2D(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                              # [B, D, n_h, n_w]
        B, D, n_h, n_w = x.shape
        x = x.view(B, D, n_h * n_w).transpose(1, 2)   # [B, P, D]
        # (no pos_embed here)
        return self.norm(x)

########################################################

class ContextEncoder2D(nn.Module):
    """
    Swin Transformer v2 Small context encoder configured for 512×512.
    Returns patch tokens (no CLS) and the token grid size (Ht, Wt).
    """
    def __init__(
        self,
        model_name: str = "swin_small_patch4_window7_224",
        pretrained: bool = True,
        img_size: int = 512,
        strict_img_size: bool = False,
        dynamic_img_pad: bool = True,
    ):
        super().__init__()
        # Build timm Swin v2; set img_size=512 and allow dynamic padding
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,          # <- run at 512
            num_classes=0,              # no classifier head
            global_pool='',             # keep token grid
            features_only=False,
            strict_img_size=strict_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.embed_dim = self.swin.num_features
        self.patch_size = 4

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]  (H,W multiples of 32 recommended; 512 works)
        returns:
          tokens: [B, P, D]  (CLS-free)
          (Ht, Wt): token grid size at the final stage (~ H/32, W/32)
        """
        feats = self.swin.forward_features(x)   # [B, L, D] or sometimes [B, Ht, Wt, D] / [B, D, Ht, Wt]

        if feats.dim() == 3:                    # [B, L, D]
            tokens = feats
            P = tokens.shape[1]
            Ht = int(math.sqrt(P))
            Wt = P // Ht
        else:
            # Handle both [B, D, Ht, Wt] and [B, Ht, Wt, D]
            if feats.shape[1] == self.embed_dim:        # [B, D, Ht, Wt]
                B, D, Ht, Wt = feats.shape
                tokens = feats.permute(0, 2, 3, 1).reshape(B, Ht * Wt, D)
            else:                                       # [B, Ht, Wt, D]
                B, Ht, Wt, D = feats.shape
                tokens = feats.reshape(B, Ht * Wt, D)

        return tokens, (Ht, Wt)

def _simple_pos_embed_2d(x: torch.Tensor) -> torch.Tensor:
    """Simple 2D positional embedding - just scaled coordinates"""
    B, D, H, W = x.shape
    device, dtype = x.device, x.dtype
    
    # Create coordinate grids
    y_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(0, 1, W, device=device, dtype=dtype) 
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Simple embedding: just use x,y coordinates repeated
    pos_embed = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1, 2, H, W]
    pos_embed = pos_embed.expand(B, -1, -1, -1)            # [B, 2, H, W]
    
    # Repeat to match channel dimension
    pos_embed = pos_embed.repeat(1, D//2, 1, 1)            # [B, D, H, W]
    if pos_embed.shape[1] < D:
        pos_embed = F.pad(pos_embed, (0, 0, 0, 0, 0, D - pos_embed.shape[1]))
    
    return pos_embed * 0.1


########################################################

class PixelDecoder2D(nn.Module):
    """
    Simple FPN-style pixel decoder. Much cleaner than deformable attention.
    Keeps same interface as the complex version.
    """
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        # Handle per-level channels (C3, C4, C5 might be different)
        in_chs: Optional[Tuple[int,int,int]] = kwargs.get("in_channels_per_level", None)
        if in_chs is None:
            in_chs = (in_channels, in_channels, in_channels)

        # Project each level to common dimension
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1),
                nn.GroupNorm(32 if embed_dim >= 32 else 1, embed_dim),
                nn.ReLU(inplace=True)
            ) for c in in_chs
        ])

        # FPN fusion layers (reduce aliasing during upsampling)
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32 if embed_dim >= 32 else 1, embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])

        # Output heads
        self.fi1_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32 if embed_dim >= 32 else 1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
        self.flast_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32 if embed_dim >= 32 else 1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )

    def forward(self, feats_multi: List[torch.Tensor], input_hw: Tuple[int, int]):
        """
        Simple FPN forward pass.
        
        feats_multi: [C3, C4, C5] at strides [1/8, 1/16, 1/32]
        input_hw: (H, W) full resolution size
        
        Returns:
            Fi1: [B, D, H/8, W/8] - feature at stride 8
            F_last: [B, D, H/4, W/4] - feature at stride 4
        """
        H, W = input_hw
        c3, c4, c5 = feats_multi
        
        # 1. Lateral connections - project to common dimension
        p5 = self.lateral_convs[2](c5)  # [B, D, H/32, W/32]
        p4 = self.lateral_convs[1](c4)  # [B, D, H/16, W/16]  
        p3 = self.lateral_convs[0](c3)  # [B, D, H/8, W/8]

        # Add positional encoding
        p5 = p5 + _simple_pos_embed_2d(p5)
        p4 = p4 + _simple_pos_embed_2d(p4)
        p3 = p3 + _simple_pos_embed_2d(p3)

        # 2. Top-down pathway (FPN fusion)
        # P5 -> P4
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = p4 + p5_up
        p4 = self.fpn_convs[1](p4)

        # P4 -> P3  
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = p3 + p4_up
        p3 = self.fpn_convs[0](p3)

        # 3. Generate outputs
        # Fi1 at stride 8 (same as p3)
        Fi1 = self.fi1_head(p3)  # [B, D, H/8, W/8]

        # F_last at stride 4 (upsample p3)
        p3_upsampled = F.interpolate(p3, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        F_last = self.flast_head(p3_upsampled)  # [B, D, H/4, W/4]

        return Fi1, F_last

class CrossAttentionBlock2D(nn.Module):
    """Cross-attention block for JEPA predictor"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, queries, features):
        # Cross-attention: queries attend to features
        attn_out, _ = self.cross_attn(queries, features, features)
        queries = self.norm1(queries + attn_out)
        
        # Feedforward
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries

class SelfAttentionBlock2D(nn.Module):
    """Self-attention block for query refinement"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward  
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, queries):
        # Self-attention: queries attend to themselves
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + attn_out)
        
        # Feedforward
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries

class Predictor2D(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_queries: int,
                 num_heads: int = None,
                 # accept both old and new arg names:
                 num_cross_blocks: int = None,
                 num_self_blocks: int = None,
                 num_cross_attn: int = None,
                 num_self_attn: int = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # choose a valid num_heads if not provided
        if num_heads is None:
            for h in (16, 12, 8, 6, 4, 3, 2, 1):
                if embed_dim % h == 0:
                    num_heads = h
                    break
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads

        # harmonize naming: prefer *attn if provided, else *blocks, else paper defaults (9/2)
        if num_cross_attn is not None:
            L = num_cross_attn
        elif num_cross_blocks is not None:
            L = num_cross_blocks
        else:
            L = 9
        if num_self_attn is not None:
            M = num_self_attn
        elif num_self_blocks is not None:
            M = num_self_blocks
        else:
            M = 2

        # learnable queries
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # Cross-attention blocks (Mask2Former-ish: norm -> cross-attn -> add, norm -> FFN -> add)
        self.cross_blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                ffn  = nn.Sequential(
                    nn.Linear(embed_dim, 4*embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(4*embed_dim, embed_dim)
                ),
                norm1 = nn.LayerNorm(embed_dim),
                norm2 = nn.LayerNorm(embed_dim),
            )) for _ in range(L)
        ])

        # Extra self-attention blocks on queries
        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(M)
        ])

        # projection head f_L to map query outputs back to Fi1 embedding space
        self.proj = nn.Linear(embed_dim, embed_dim)

        # learnable mask token for masked Fi1 tiles (used to build K/V when Fi1 is masked)
        self.kv_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.kv_mask_token, std=0.02)

    @staticmethod
    def _add_2d_sincos_pos(feat_2d: torch.Tensor):
        """
        feat_2d: [B, D, H, W] -> returns [B, H*W, D] with fixed 2D sin/cos pos enc added
        """
        import torch.nn.functional as F
        B, D, H, W = feat_2d.shape
        device = feat_2d.device
    
        # build sincos grid
        y = torch.linspace(-1, 1, steps=H, device=device)
        x = torch.linspace(-1, 1, steps=W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack([xx, yy], dim=-1).reshape(1, H, W, 2)  # [1,H,W,2]
    
        # project to channel dim D using sin/cos
        half = D // 2
        sin_in = pos[..., 0:1].repeat(1, 1, 1, half)
        cos_in = pos[..., 1:2].repeat(1, 1, 1, D - half)
        pos_embed = torch.cat([torch.sin(sin_in), torch.cos(cos_in)], dim=-1)  # [1,H,W,D]
        if pos_embed.shape[-1] != D:
            pos_embed = F.pad(pos_embed, (0, D - pos_embed.shape[-1]))[:, :, :, :D]
    
        # >>> key fix: match [B, D, H, W] before addition
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, D, H, W]
    
        feat = feat_2d + pos_embed.to(feat_2d.dtype)  # [B, D, H, W]
        feat = feat.flatten(2).transpose(1, 2)        # [B, H*W, D]
        return feat

    def forward(self, Fi1_online: torch.Tensor, Fi1_mask: torch.Tensor):
        """
        Fi1_online: [B, D, H8, W8]   (online pixel-decoder Fi1)
        Fi1_mask:   [B, H8*W8] bool  (True=masked positions in Fi1 to reconstruct)
        Returns:
           pred_masked_feats: [B, M, D] predictions for masked Fi1 positions (M = #masked)
           masked_indices:    [B, M] indices of the masked Fi1 positions (or -1 padded)
           query_feats:       [B, Q, D] final query embeddings
        """
        B, D, H8, W8 = Fi1_online.shape
        Q = self.num_queries

        # Build K/V from Fi1 with 2D sincos pos; replace masked tiles with kv_mask_token (NOT zeros)
        kv = Fi1_online.clone()  # [B,D,H8,W8]
        if Fi1_mask is not None:
            mask_2d = Fi1_mask.reshape(B, H8, W8).unsqueeze(1).expand(-1, D, -1, -1)  # [B,D,H8,W8]
            kv = torch.where(mask_2d, self.kv_mask_token.view(1, D, 1, 1).expand(B, D, H8, W8), kv)

        kv_seq = self._add_2d_sincos_pos(kv)  # [B, H8*W8, D] as K/V

        # Queries
        q = self.query_embed.expand(B, Q, D)  # [B,Q,D]

        # L cross-attention blocks
        for blk in self.cross_blocks:
            qn = blk['norm1'](q)
            attn_out, _ = blk['attn'](qn, kv_seq, kv_seq)  # cross-attn to Fi1+pos
            q = q + attn_out
            q = q + blk['ffn'](blk['norm2'](q))

        # M self-attn blocks on queries
        for sblk in self.self_blocks:
            q = sblk(q)

        # map to Fi1 embedding space
        q_proj = self.proj(q)  # [B,Q,D]

        # Route query embeddings to masked tiles
        scores = torch.einsum('bpd,bqd->bpq', kv_seq, q_proj) / (D ** 0.5)  # [B,P,Q]
        probs = scores.softmax(dim=-1)  # over queries

        if Fi1_mask is not None and Fi1_mask.any():
            pred_list, idx_list = [], []
            for b in range(B):
                mask_b = Fi1_mask[b]  # [P]
                if mask_b.any():
                    prob_b = probs[b, mask_b]     # [Mb, Q]
                    q_b    = q_proj[b]            # [Q, D]
                    pred_b = prob_b @ q_b         # [Mb, D]
                    pred_list.append(pred_b)
                    idx_list.append(mask_b.nonzero(as_tuple=False).squeeze(1))
                else:
                    pred_list.append(q_proj.new_zeros((0, D)))
                    idx_list.append(torch.zeros((0,), dtype=torch.long, device=q_proj.device))
            maxM = max([p.size(0) for p in pred_list])
            if maxM == 0:
                pred_masked_feats = q_proj.new_zeros((B, 0, D))
                masked_indices    = q_proj.new_zeros((B, 0), dtype=torch.long)
            else:
                pred_masked_feats, masked_indices = [], []
                for b in range(B):
                    m = pred_list[b].size(0)
                    pad = maxM - m
                    if pad > 0:
                        pred_masked_feats.append(F.pad(pred_list[b], (0,0,0,pad)))
                        masked_indices.append(F.pad(idx_list[b], (0,pad), value=-1))
                    else:
                        pred_masked_feats.append(pred_list[b])
                        masked_indices.append(idx_list[b])
                pred_masked_feats = torch.stack(pred_masked_feats, dim=0)  # [B, M, D]
                masked_indices    = torch.stack(masked_indices, dim=0)     # [B, M]
        else:
            pred_masked_feats = q_proj.new_zeros((B, 0, D))
            masked_indices    = q_proj.new_zeros((B, 0), dtype=torch.long)

        return pred_masked_feats, masked_indices, q_proj


########################################################
# Enhanced Segmentation Head Components
########################################################

def _gn_groups(C):
    """Get optimal GroupNorm groups for given channels"""
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0: 
            return g
    return 1


class DWSepResBlock(nn.Module):
    """Depthwise Separable Residual Block with Dilated Convolution"""
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                           groups=channels, bias=False)
        self.dw_g = nn.GroupNorm(_gn_groups(channels), channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.pw_g = nn.GroupNorm(_gn_groups(channels), channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.act(self.dw_g(self.dw(x)))
        y = self.pw_g(self.pw(y))
        y = self.dropout(y)  # Spatial dropout
        return self.act(x + y)


class EnhancedSpatialGate(nn.Module):
    """Enhanced Spatial Attention with better processing"""
    def __init__(self, channels: int):
        super().__init__()
        mid_ch = max(1, channels // 4)
        self.conv1 = nn.Conv2d(channels, mid_ch, 3, padding=1, groups=min(channels, 8), bias=False)
        self.norm1 = nn.GroupNorm(_gn_groups(mid_ch), mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        attn = self.act(self.norm1(self.conv1(x)))
        attn = torch.sigmoid(self.conv2(attn))
        return x * attn


class EnhancedChannelSE(nn.Module):
    """Enhanced Channel Squeeze-Excitation with multiple ratios"""
    def __init__(self, channels: int, reduction_ratios: List[int] = [4, 8, 16]):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in reduction_ratios:
            mid_ch = max(1, channels // r)
            branch = nn.Sequential(
                nn.Conv2d(channels, mid_ch, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, channels, 1, bias=True)
            )
            self.branches.append(branch)
        
        # Learnable weights
        self.combine_weights = nn.Parameter(torch.ones(len(reduction_ratios)))

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        
        # Multi-branch processing
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(s))
        
        # Weighted combination
        weights = torch.softmax(self.combine_weights, dim=0)
        combined = sum(w * out for w, out in zip(weights, branch_outputs))
        
        return x * torch.sigmoid(combined)


class GlobalContextModule(nn.Module):
    """Global Average Pooling branch for scene-level context"""
    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_gn_groups(channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Global context
        global_feat = self.gap(x)  # [B, C, 1, 1]
        global_feat = self.conv(global_feat)  # Process
        global_feat = global_feat.expand(B, C, H, W)  # Broadcast back
        return global_feat


class SegmentationHead(nn.Module):
    """
    Enhanced segmentation head with:
    - 256 mid_channels for more capacity
    - ASPP-style multi-scale dilations [1, 6, 12, 18] 
    - 5 refinement blocks for deeper processing
    - Global context branch for scene understanding
    - Enhanced dual attention (spatial + channel)
    - Learnable fusion weights
    - Spatial dropout for regularization
    """
    def __init__(self, 
                 in_channels: int, 
                 mid_channels: int = 256, 
                 num_classes: int = 150,
                 dropout: float = 0.1,
                 fi1_channels: int = None):
        super().__init__()
        
        # Feature reduction with learnable fusion
        fi1_in_channels = fi1_channels if fi1_channels is not None else in_channels
        self.fi1_reduce = nn.Sequential(
            nn.Conv2d(fi1_in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
        )
        self.flast_reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),  # flast uses main in_channels
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0]))  # [fi1_weight, flast_weight]
        
        # Initial fusion
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Global context
        self.global_context = GlobalContextModule(mid_channels)
        
        # ASPP-style refinement blocks with different dilations
        dilations = [1, 6, 12, 18, 1]  # 5 blocks as planned
        self.refinement_blocks = nn.ModuleList([
            DWSepResBlock(mid_channels, dilation=dil, dropout=dropout) 
            for dil in dilations
        ])
        
        # Dual attention
        self.spatial_attn = EnhancedSpatialGate(mid_channels)
        self.channel_attn = EnhancedChannelSE(mid_channels)
        
        # Classification head
        self.classifier = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, fi1, flast, out_hw):
        """
        Args:
            fi1: [B, D, H/8, W/8] - Features at stride 8
            flast: [B, D, H/4, W/4] - Features at stride 4  
            out_hw: (H, W) - Target output size
        
        Returns:
            logits: [B, num_classes, H, W] - Segmentation logits
        """
        H, W = out_hw
        
        # Upsample Fi1 to match F_last spatial resolution
        fi1_up = F.interpolate(fi1, size=flast.shape[-2:], mode='bilinear', align_corners=False)
        
        # Feature reduction
        fi1_reduced = self.fi1_reduce(fi1_up)      # [B, 256, H/4, W/4]
        flast_reduced = self.flast_reduce(flast)   # [B, 256, H/4, W/4]
        
        # Learnable weighted fusion
        fusion_weights = torch.softmax(self.fusion_weights, dim=0)
        fused = fusion_weights[0] * fi1_reduced + fusion_weights[1] * flast_reduced
        fused = self.fuse_conv(fused)  # [B, 256, H/4, W/4]
        
        # Add global context
        global_context = self.global_context(fused)
        fused = fused + global_context
        
        # Multi-scale refinement with ASPP dilations
        x = fused
        for block in self.refinement_blocks:
            x = block(x)
        
        # Dual attention
        x = self.spatial_attn(x)
        x = self.channel_attn(x)
        
        # ADD THIS LINE - Dropout for regularization
        x = F.dropout(x, p=0.4, training=self.training)
        
        # Final classification
        logits = self.classifier(x)  # [B, num_classes, H/4, W/4]
        
        # Upsample to target resolution
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits  # [B, num_classes, H, W]

########################################################
# UPerNet - Unified Perceptual Parsing for Scene Understanding
########################################################

class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module for global context aggregation"""
    def __init__(self, in_channels, out_channels=512, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.pool_scales = pool_scales
        
        # Pool branches
        self.pool_branches = nn.ModuleList()
        for scale in pool_scales:
            self.pool_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels // len(pool_scales), 1, bias=False),
                nn.GroupNorm(_gn_groups(out_channels // len(pool_scales)), out_channels // len(pool_scales)),
                nn.ReLU(inplace=True)
            ))
        
        # Final conv
        total_out = out_channels + in_channels  # Concat with original
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_out, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - Input features
        Returns:
            [B, out_channels, H, W] - Enhanced features with global context
        """
        h, w = x.shape[-2:]
        pool_outs = []
        
        # Global average pooling
        for i, pool_branch in enumerate(self.pool_branches):
            pooled = pool_branch(x)
            # Upsample to original size
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pool_outs.append(pooled)
        
        # Concatenate all branches
        out = torch.cat([x] + pool_outs, dim=1)
        return self.final_conv(out)

class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.GroupNorm(_gn_groups(out_channels), out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_gn_groups(out_channels), out_channels), 
                nn.ReLU(inplace=True)
            ))

    def forward(self, inputs):
        """
        Args:
            inputs: List of feature maps from backbone [low_res -> high_res]
                   e.g., [f1: [B,C1,H/32,W/32], f2: [B,C2,H/16,W/16], f3: [B,C3,H/8,W/8], f4: [B,C4,H/4,W/4]]
        Returns:
            List of FPN features at different scales
        """
        # Build laterals (reduce channel dimensions)
        laterals = [
            lateral_conv(inputs[i]) 
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher-level feature map
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            # Element-wise addition
            laterals[i] = laterals[i] + upsampled
        
        # Final convolutions
        outs = [
            self.fpn_convs[i](laterals[i]) 
            for i in range(len(laterals))
        ]
        
        return outs

class UPerNetHead(nn.Module):
    """
    UPerNet Head - PROVEN Architecture for ViT + ADE20K
    
    Features:
    Multi-scale backbone feature extraction (4 stages)
    Pyramid Pooling Module for global context  
    Feature Pyramid Network for multi-scale fusion
    Progressive upsampling to target resolution
    Optimized for SwinV2-Small on ADE20K
    """
    def __init__(self, 
                 in_channels_list: List[int],  # e.g., [96, 192, 384, 768] for SwinV2-Small
                 num_classes: int = 150,
                 fpn_out_channels: int = 256,
                 ppm_out_channels: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes
        
        # Pyramid Pooling Module on highest-level features
        self.ppm = PyramidPoolingModule(
            in_channels=in_channels_list[-1],  # Deepest features
            out_channels=ppm_out_channels
        )
        
        # Update channels list
        fpn_in_channels = in_channels_list[:-1] + [ppm_out_channels]
        
        # Feature Pyramid Network decoder
        self.fpn_decoder = FPNDecoder(
            in_channels_list=fpn_in_channels,
            out_channels=fpn_out_channels
        )
        
        # Final fusion
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fpn_out_channels * len(fpn_in_channels), fpn_out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(fpn_out_channels), fpn_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        self.classifier = nn.Conv2d(fpn_out_channels, num_classes, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following UPerNet paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            features: List of multi-scale features from backbone [low->high resolution]
                     e.g., [f1: [B,192,H/32,W/32], f2: [B,192,H/16,W/16], 
                            f3: [B,192,H/8,W/8], f4: [B,192,H/4,W/4]]
            out_hw: Target output size (H, W)
            
        Returns:
            logits: [B, num_classes, H, W] - Segmentation predictions
        """
        H, W = out_hw
        
        # Apply Pyramid Pooling to deepest features for global context
        features_with_ppm = features[:-1] + [self.ppm(features[-1])]
        
        # FPN decoder - multi-scale feature fusion
        fpn_outs = self.fpn_decoder(features_with_ppm)
        
        # Upsample all FPN outputs to same resolution (1/4 scale) - STANDARD UPerNet
        target_size = (H // 4, W // 4)  # STANDARD: Use 1/4 resolution like original UPerNet
        upsampled_fpn = []
        for fpn_out in fpn_outs:
            upsampled = F.interpolate(
                fpn_out, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            upsampled_fpn.append(upsampled)
        
        # Concatenate and fuse scales
        fused_features = torch.cat(upsampled_fpn, dim=1)  # [B, fpn_out_channels * num_scales, H/4, W/4]
        fused_features = self.fuse_conv(fused_features)   # [B, fpn_out_channels, H/4, W/4]
        
        # Generate predictions
        logits = self.classifier(fused_features)  # [B, num_classes, H/4, W/4]
        
        # Upsample to final target resolution
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits  # [B, num_classes, H, W]