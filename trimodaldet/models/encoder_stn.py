"""
STN-augmented dual-stream backbone variant for tri-modal detection.

Injects STNAlignBlock between per-stage transformer output (after LayerNorm
and reshape to BCHW) and the MAGE cross-modal fusion.  The STN learns a
small affine delta on top of hard-coded calibration priors to compensate
for residual sensor misalignment.

Usage:
    from trimodaldet.models.encoder_stn import mit_b1_stn, get_encoder_stn

    encoder = get_encoder_stn('mit_b1_stn', in_chans_rgb=3, in_chans_x=2)
    outs = encoder.forward_features(rgb, x)   # same interface as InterModalBackbone
"""

import torch
import torch.nn as nn
from functools import partial

from .encoder import InterModalBackbone, mit_b0, mit_b1, mit_b2, mit_b3, mit_b4
from .transformer import Block, OverlapPatchEmbed
from .fusion import MAGE, BiTE
from ..alignment.stn_align import STNAlignBlock, DEFAULT_IR_TO_RGB, DEFAULT_RGB_TO_IR


class InterModalBackboneSTN(InterModalBackbone):
    """
    STN-augmented variant.  Identical to InterModalBackbone except that
    between reshape→BCHW and MAGE at each stage, STNAlignBlock is applied.

    Extra Parameters
    ----------------
    stn_reduction : int, default 32
        Channel reduction ratio for the STN regressor bottleneck.
    stn_enabled_stages : list[int] | None
        Which stages (0–3) apply STN.  None = all four.
    stn_theta_ab : Tensor | None
        2×3 affine prior for modality-A → modality-B.  If None, uses
        DEFAULT_IR_TO_RGB (from MM-UAV calibration).
    stn_theta_ba : Tensor | None
        2×3 affine prior for modality-B → modality-A.
    """

    def __init__(self, img_size=224, in_chans_rgb=3, in_chans_x=2,
                 embed_dims=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8), mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 norm_fuse=nn.BatchNorm2d, depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 stn_reduction=32,
                 stn_enabled_stages=None,
                 stn_theta_ab=None,
                 stn_theta_ba=None):
        super().__init__(img_size, in_chans_rgb, in_chans_x, 1000,
                         embed_dims, num_heads, mlp_ratios, qkv_bias,
                         qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                         norm_layer, norm_fuse, depths, sr_ratios)

        if stn_enabled_stages is None:
            stn_enabled_stages = [0, 1, 2, 3]

        if stn_theta_ab is None:
            stn_theta_ab = DEFAULT_IR_TO_RGB.clone()
        if stn_theta_ba is None:
            stn_theta_ba = DEFAULT_RGB_TO_IR.clone()

        self.stn_blocks = nn.ModuleList([
            STNAlignBlock(
                channels=embed_dims[i],
                reduction=stn_reduction,
                default_theta_ab=stn_theta_ab,
                default_theta_ba=stn_theta_ba,
                enable=(i in stn_enabled_stages))
            for i in range(4)
        ])

    # ------------------------------------------------------------------
    def _process_stage(self, stage, x_rgb, x_x, B, H, W):
        """
        norm → reshape (B,N,C) → (B,C,H,W) → STN → MAGE → BiTE → fused
        """
        norm_rgb = getattr(self, f'norm{stage + 1}')
        norm_x  = getattr(self, f'extra_norm{stage + 1}')
        x_rgb, x_x = norm_rgb(x_rgb), norm_x(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x   = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ── STN injection point ──
        x_rgb, x_x = self.stn_blocks[stage](x_rgb, x_x)

        x_rgb, x_x = self.MAGEs[stage](x_rgb, x_x)
        fused = self.BiTEs[stage](x_rgb, x_x)
        return x_rgb, x_x, fused

    # ------------------------------------------------------------------
    def forward_features(self, x_rgb, x_x):
        B = x_rgb.shape[0]
        outs = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_x,   _, _ = self.extra_patch_embed1(x_x)
        for blk in self.block1:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1: x_x   = blk(x_x, H, W)
        x_rgb, x_x, fused = self._process_stage(0, x_rgb, x_x, B, H, W)
        outs.append(fused)

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_x,   _, _ = self.extra_patch_embed2(x_x)
        for blk in self.block2:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2: x_x   = blk(x_x, H, W)
        x_rgb, x_x, fused = self._process_stage(1, x_rgb, x_x, B, H, W)
        outs.append(fused)

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_x,   _, _ = self.extra_patch_embed3(x_x)
        for blk in self.block3:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3: x_x   = blk(x_x, H, W)
        x_rgb, x_x, fused = self._process_stage(2, x_rgb, x_x, B, H, W)
        outs.append(fused)

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_x,   _, _ = self.extra_patch_embed4(x_x)
        for blk in self.block4:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4: x_x   = blk(x_x, H, W)
        x_rgb, x_x, fused = self._process_stage(3, x_rgb, x_x, B, H, W)
        outs.append(fused)

        return outs


# ── Concrete STN backbone variants  (match mit_b0 … mit_b4 signatures) ──

class mit_b0_stn(InterModalBackboneSTN):
    def __init__(self, in_chans_rgb=3, in_chans_x=2, **kwargs):
        super().__init__(
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.1,
            in_chans_rgb=in_chans_rgb,
            in_chans_x=in_chans_x,
            **kwargs)


class mit_b1_stn(InterModalBackboneSTN):
    def __init__(self, in_chans_rgb=3, in_chans_x=2, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.1,
            in_chans_rgb=in_chans_rgb,
            in_chans_x=in_chans_x,
            **kwargs)


class mit_b2_stn(InterModalBackboneSTN):
    def __init__(self, in_chans_rgb=3, in_chans_x=2, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.1,
            in_chans_rgb=in_chans_rgb,
            in_chans_x=in_chans_x,
            **kwargs)


class mit_b3_stn(InterModalBackboneSTN):
    def __init__(self, in_chans_rgb=3, in_chans_x=2, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.1,
            in_chans_rgb=in_chans_rgb,
            in_chans_x=in_chans_x,
            **kwargs)


class mit_b4_stn(InterModalBackboneSTN):
    def __init__(self, in_chans_rgb=3, in_chans_x=2, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.1,
            in_chans_rgb=in_chans_rgb,
            in_chans_x=in_chans_x,
            **kwargs)


STN_BACKBONE_REGISTRY = {
    'mit_b0_stn': mit_b0_stn,
    'mit_b1_stn': mit_b1_stn,
    'mit_b2_stn': mit_b2_stn,
    'mit_b3_stn': mit_b3_stn,
    'mit_b4_stn': mit_b4_stn,
}


def get_encoder_stn(backbone_name, in_chans_rgb=3, in_chans_x=2, **stn_kwargs):
    """Factory for STN encoder variants.  Mirrors get_encoder()."""
    if backbone_name not in STN_BACKBONE_REGISTRY:
        available = ', '.join(sorted(STN_BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown STN backbone '{backbone_name}'. Available: {available}")
    encoder_class = STN_BACKBONE_REGISTRY[backbone_name]
    return encoder_class(in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
                         **stn_kwargs)


__all__ = ['InterModalBackboneSTN',
           'mit_b0_stn', 'mit_b1_stn', 'mit_b2_stn', 'mit_b3_stn', 'mit_b4_stn',
           'STN_BACKBONE_REGISTRY', 'get_encoder_stn']