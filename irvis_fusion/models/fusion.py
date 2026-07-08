from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SAMQKVFusion(nn.Module):
    """Small-target-aware SAM-QKV cross-modal attention.

    SAM is embedded as a soft spatial prior before Q/K/V projection. Attention
    combines local-window contrast and global context so sparse distant targets
    are not suppressed by a full-image softmax. A lightweight modality gate
    increases IR contribution in locally salient thermal regions.
    """

    def __init__(self, channels: int, local_windows: tuple[int, ...] = (3, 7, 11)) -> None:
        super().__init__()
        self.scale = channels ** -0.5
        self.local_windows = local_windows
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sam_embed = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.local_global_mix = nn.Parameter(torch.tensor(1.0))
        self.ir_prior_strength = nn.Parameter(torch.tensor(0.5))
        self.modality_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.base_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.norm=nn.InstanceNorm2d(channels)

    def _hybrid_attention(self, score: torch.Tensor) -> torch.Tensor:
        local_maps = []
        for window in self.local_windows:
            local_score = F.max_pool2d(
                score,
                kernel_size=window,
                stride=1,
                padding=window // 2,
            )
            local_maps.append(torch.sigmoid(local_score))

        local_attn = torch.stack(local_maps, dim=0).mean(dim=0)
        global_attn = torch.sigmoid(score.mean(dim=(-2, -1), keepdim=True))
        local_weight = torch.sigmoid(self.local_global_mix)

        return local_weight * local_attn + (1.0 - local_weight) * global_attn

    def _ir_modality_weight(
        self,
        ir_cond: torch.Tensor,
        vis_cond: torch.Tensor,
        sam_focus: torch.Tensor | None,
    ) -> torch.Tensor:
        ir_energy = ir_cond.abs().mean(dim=1, keepdim=True)
        vis_energy = vis_cond.abs().mean(dim=1, keepdim=True)
        diff_energy = (ir_cond - vis_cond).abs().mean(dim=1, keepdim=True)
        if sam_focus is None:
            sam_focus = ir_energy.new_zeros(ir_energy.shape)

        gate_logits = self.modality_gate(
            torch.cat([ir_energy, vis_energy, diff_energy, sam_focus], dim=1)
        )
        # 【修改点】移除了 local_ir_mean 和 local_ir_saliency，避免边缘过度加权
        thermal_advantage = torch.tanh(ir_energy - vis_energy)
        ir_bias = self.ir_prior_strength * thermal_advantage
        return torch.sigmoid(gate_logits + ir_bias)

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, h, w = ir.shape
        if sam_attention is not None:
            sam = F.interpolate(
                sam_attention,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            sam_prior = self.sam_embed(sam) * sam
        else:
            sam = None
            sam_prior = ir.new_zeros(ir.shape)

        ir_cond = ir + sam_prior
        vis_cond = vis + sam_prior

        q_ir = self.q_proj(ir_cond)
        k_vis = self.k_proj(vis_cond)
        v_vis = self.v_proj(vis_cond)

        q_vis = self.q_proj(vis_cond)
        k_ir = self.k_proj(ir_cond)
        v_ir = self.v_proj(ir_cond)

        score_ir_to_vis = (q_ir * k_vis).sum(dim=1, keepdim=True) * self.scale
        score_vis_to_ir = (q_vis * k_ir).sum(dim=1, keepdim=True) * self.scale
        attn_ir_to_vis = self._hybrid_attention(score_ir_to_vis)
        attn_vis_to_ir = self._hybrid_attention(score_vis_to_ir)

        ir_weight = self._ir_modality_weight(ir_cond, vis_cond, sam)
        cross_vis = attn_ir_to_vis * v_vis
        cross_ir = attn_vis_to_ir * v_ir
        base = self.base_proj(ir_weight * ir_cond + (1.0 - ir_weight) * vis_cond)
        fused = base + ir_weight * cross_ir + (1.0 - ir_weight) * cross_vis
        fused = self.norm(fused)
        return self.out(fused)


class SAMQKVUnifiedFusion(nn.Module):
    """Three-scale wrapper around SAM-QKV fusion.

    F_fuse = f(F_ir, F_vis, A_sam)

    Detection feedback is intentionally absent from this forward path. It is
    handled by the training objective as a dynamic detection-loss weight.
    """

    def __init__(self, channels: int = 128, num_scales: int = 3) -> None:
        super().__init__()
        self.operators = nn.ModuleList(
            [SAMQKVFusion(channels) for _ in range(num_scales)]
        )

    def forward(
        self,
        ir_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        vis_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sam_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(ir_features) != len(vis_features) or len(ir_features) != len(self.operators):
            raise ValueError("SAMQKVUnifiedFusion expects three aligned FPN scales.")
        fused = [
            operator(ir, vis, sam_attention)
            for operator, ir, vis in zip(self.operators, ir_features, vis_features)
        ]
        return tuple(fused)  # type: ignore[return-value]