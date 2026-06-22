"""
STN (Spatial Transformer Network) Feature-Level Alignment for Tri-Modal Detection.

Adapted from MM-UAV-Benchmark's YOLOPAFPN2 STN fusion (yolo_pafpn2_stn.py).
Aligns cross-modal features BEFORE MAGE/BiTE fusion via learned affine
transforms with hard-coded calibration priors.

Principle:
  - Each modality pair (RGB↔Thermal, RGB↔Event) has a frozen default 2x3
    affine matrix representing the pre-calibrated sensor geometry.
  - A lightweight CNN regressor predicts a small delta (scaled by 0.001)
    from concatenated modality features.
  - F.affine_grid + F.grid_sample applies the spatial transform with
    bilinear interpolation and zero-padding.

This module operates at a single FPN scale. Instantiate one per scale
(e.g., for 80x80, 40x40, 20x20 features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Default affine matrices from MM-UAV-Benchmark (IR↔RGB calibration) ──
# These are frozen priors; the STN learns a small delta on top.
#
# IR -> RGB: scales IR features down by ~1.216x and translates to align
# with RGB coordinate frame.  Derived exogenously from sensor geometry.
DEFAULT_IR_TO_RGB = torch.tensor(
    [[1.2161, -0.0351, -0.0825],
     [0.0351,  1.2161,  0.2926]],
    dtype=torch.float)

DEFAULT_RGB_TO_IR = torch.tensor(
    [[0.8271,  0.0119,  0.0615],
     [-0.0119, 0.8271, -0.2396]],
    dtype=torch.float)


class STNAlignBlock(nn.Module):
    """
    Aligns two modality feature maps using an affine STN.

    Args:
        channels: Number of channels in the input feature maps.
        reduction: Channel reduction ratio for the STN regressor bottleneck.
                   The regressor operates on concatenated features (2*channels)
                   after reducing to `channels // reduction` each.
        default_theta_ab:  2x3 tensor — default affine from modality A to B.
        default_theta_ba:  2x3 tensor — default affine from modality B to A.
        enable: If False, behaves as identity (for modality ablation).
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 32,
        default_theta_ab: torch.Tensor | None = None,
        default_theta_ba: torch.Tensor | None = None,
        enable: bool = True,
    ):
        super().__init__()
        self.enable = enable
        c_in = channels // reduction  # bottleneck after channel reduction conv

        if not enable:
            return

        # Channel reduction for STN input (each modality: C -> C/reduction)
        self.conv_reduce = nn.Conv2d(channels, c_in, kernel_size=3, padding=1)

        # Adaptive pooling to fixed spatial size before final conv
        self.stn = nn.Sequential(
            nn.Conv2d(2 * c_in, c_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),

            nn.Conv2d(c_in, c_in * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in * 2, c_in * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(c_in * 4, 6),
            nn.Tanh(),
        )

        # Zero-init the final linear layer
        self.stn[-2].weight.data.zero_()
        self.stn[-2].bias.data.zero_()

        # Register default matrices as non-trainable parameters
        if default_theta_ab is not None:
            self.register_buffer('default_theta_ab', default_theta_ab.clone())
        else:
            self.register_buffer('default_theta_ab', torch.eye(2, 3))

        if default_theta_ba is not None:
            self.register_buffer('default_theta_ba', default_theta_ba.clone())
        else:
            self.register_buffer('default_theta_ba', torch.eye(2, 3))

        # Post-alignment channel adaptation
        self.conv_post_ab = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_post_ba = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """
        Args:
            feat_a: Feature map from modality A  (N, C, H, W)
            feat_b: Feature map from modality B  (N, C, H, W)

        Returns:
            feat_a_aligned: feat_a aligned to B's coordinate frame
            feat_b_aligned: feat_b aligned to A's coordinate frame
        """
        if not self.enable:
            return feat_a, feat_b

        N, C, H, W = feat_a.shape

        # ── Step 1: Channel reduction ──
        a_red = self.conv_reduce(feat_a)  # (N, C//R, H, W)
        b_red = self.conv_reduce(feat_b)

        # ── Step 2: Predict delta theta ──
        # a->b: regress from [a_red, b_red]
        delta_ab = self.stn(torch.cat([a_red, b_red], dim=1)).view(N, 2, 3)
        # b->a: regress from [b_red, a_red]
        delta_ba = self.stn(torch.cat([b_red, a_red], dim=1)).view(N, 2, 3)

        # ── Step 3: Final theta = default + delta * 0.001 ──
        theta_ab = self.default_theta_ab.to(feat_a.device).unsqueeze(0).repeat(N, 1, 1) \
                   + delta_ab * 0.001
        theta_ba = self.default_theta_ba.to(feat_a.device).unsqueeze(0).repeat(N, 1, 1) \
                   + delta_ba * 0.001

        # ── Step 4: Spatial transform ──
        grid_ab = F.affine_grid(theta_ab, [N, C, H, W], align_corners=True)
        grid_ba = F.affine_grid(theta_ba, [N, C, H, W], align_corners=True)

        feat_a_aligned = F.grid_sample(
            feat_a, grid_ab, mode='bilinear',
            padding_mode='zeros', align_corners=True)
        feat_b_aligned = F.grid_sample(
            feat_b, grid_ba, mode='bilinear',
            padding_mode='zeros', align_corners=True)

        # ── Step 5: Post-alignment channel adaptation ──
        feat_a_aligned = self.conv_post_ab(feat_a_aligned)
        feat_b_aligned = self.conv_post_ba(feat_b_aligned)

        return feat_a_aligned, feat_b_aligned


class AdaptiveWeightFusion(nn.Module):
    """
    Channel-wise adaptive fusion with residual connection.

    Computes per-channel attention weights from the global average pool
    of the concatenated feature pair, then produces a weighted sum:
        fused = alpha * feat_a + beta * feat_b
    with a residual: output = fused + feat_a.

    This is identical to the AdaptiveWeightFusion used in MM-UAV YOLOPAFPN2.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 2 * channels),
            nn.Sigmoid(),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        B, C = feat_a.size(0), feat_a.size(1)
        avg_a = self.avg_pool(feat_a).view(B, C)
        avg_b = self.avg_pool(feat_b).view(B, C)
        combined = torch.cat([avg_a, avg_b], dim=1)
        weights = self.fc(combined).view(B, 2 * C, 1, 1)
        w_a, w_b = torch.split(weights, C, dim=1)
        return feat_a * w_a + feat_b * w_b


class STNFusionModule(nn.Module):
    """
    Full STN alignment + adaptive fusion module for one FPN scale.

    Flow:
        1. STNAlignBlock: cross-warp feat_a ↔ feat_b
        2. AdaptiveWeightFusion (a->b direction): fuse A + aligned_B
        3. AdaptiveWeightFusion (b->a direction): fuse B + aligned_A
        4. Residual connection on each direction

    Args:
        channels: Feature channels at this scale.
        default_theta_ab: 2x3 affine prior for A→B alignment.
        default_theta_ba: 2x3 affine prior for B→A alignment.
        stn_enable: If False, alignment is skipped (fallback to direct fusion).
    """

    def __init__(
        self,
        channels: int,
        default_theta_ab: torch.Tensor | None = None,
        default_theta_ba: torch.Tensor | None = None,
        stn_enable: bool = True,
    ):
        super().__init__()
        self.stn_enable = stn_enable

        if stn_enable:
            self.align = STNAlignBlock(
                channels=channels,
                default_theta_ab=default_theta_ab,
                default_theta_ba=default_theta_ba,
                enable=True,
            )

        self.fusion_ab = AdaptiveWeightFusion(channels)
        self.fusion_ba = AdaptiveWeightFusion(channels)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """
        Args:
            feat_a: Modality A features  (N, C, H, W)
            feat_b: Modality B features  (N, C, H, W)

        Returns:
            fused_a: A-domain fused features
            fused_b: B-domain fused features
        """
        if self.stn_enable:
            a_aligned, b_aligned = self.align(feat_a, feat_b)
        else:
            a_aligned, b_aligned = feat_a, feat_b

        # Fuse in each coordinate frame + residual
        fused_a = self.fusion_ab(feat_a, b_aligned) + feat_a
        fused_b = self.fusion_ba(feat_b, a_aligned) + feat_b

        return fused_a, fused_b


# ── Three-modality convenience: multi-pair alignments ──

class TriModalSTNAlign(nn.Module):
    """
    Manages STN alignment for three modalities (RGB, Thermal, Event).

    Provides three pair-wise STNFusionModules:
        - RGB ↔ Thermal  (with MM-UAV calibration priors)
        - RGB ↔ Event    (default: identity prior, since no calibration exists)
        - Thermal ↔ Event (default: identity prior)

    Usage:
        stn = TriModalSTNAlign(channels=[256, 256, 256])  # one per FPN scale
        ...
        rgb_aligned_fused, thermal_fused, event_fused = stn(
            [rgb_feat, thermal_feat, event_feat], fpn_level)
    """

    def __init__(
        self,
        channels: list[int],
        enable: bool = True,
    ):
        super().__init__()
        self.enable = enable
        assert len(channels) >= 1, "Need at least one FPN scale"

        if not enable:
            return

        identity_2x3 = torch.eye(2, 3)

        self.fusion_rt = nn.ModuleList([
            STNFusionModule(
                channels=c,
                default_theta_ab=DEFAULT_IR_TO_RGB.clone(),
                default_theta_ba=DEFAULT_RGB_TO_IR.clone(),
                stn_enable=True,
            )
            for c in channels
        ])

        # For RGB↔Event and Thermal↔Event, no calibration prior exists.
        # Use identity matrix as default — the STN learns alignment from scratch
        # if needed, or just passes through with adaptive fusion.
        self.fusion_re = nn.ModuleList([
            STNFusionModule(c, identity_2x3.clone(), identity_2x3.clone(), stn_enable=True)
            for c in channels
        ])

        self.fusion_te = nn.ModuleList([
            STNFusionModule(c, identity_2x3.clone(), identity_2x3.clone(), stn_enable=True)
            for c in channels
        ])

    def forward(self, rgb_feats: list[torch.Tensor],
                thermal_feats: list[torch.Tensor],
                event_feats: list[torch.Tensor],
                level: int = 0):
        """
        Args:
            rgb_feats: List of RGB feature maps per FPN scale.
            thermal_feats: List of Thermal feature maps per FPN scale.
            event_feats: List of Event feature maps per FPN scale.
            level: Which FPN scale index to process.

        Returns:
            rgb_fused, thermal_fused, event_fused at the given level.
        """
        if not self.enable:
            return rgb_feats[level], thermal_feats[level], event_feats[level]

        # Pair-wise fusion: RGB↔Thermal, RGB↔Event, Thermal↔Event
        # Strategy: fuse each pair independently, then combine results
        # via element-wise mean or learned combination.
        #
        # For now: apply RGB↔Thermal STN first, then RGB↔Event STN,
        # using the fused RGB as the anchor.

        rt_rgb, rt_thermal = self.fusion_rt[level](rgb_feats[level], thermal_feats[level])
        re_rgb, re_event = self.fusion_re[level](rt_rgb, event_feats[level])
        te_thermal, te_event = self.fusion_te[level](rt_thermal, re_event)

        return re_rgb, te_thermal, te_event

    def get_stn_params(self):
        """Return all STN regressor parameters for separate LR scheduling."""
        params = []
        for block_list in [self.fusion_rt, self.fusion_re, self.fusion_te]:
            for block in block_list:
                if block.stn_enable:
                    params.extend(block.align.stn.parameters())
        return params