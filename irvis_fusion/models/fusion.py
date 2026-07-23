from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelCrossAttention(nn.Module):
    """通道级跨模态注意力（带大特征图自动降采样）。

    对于空间尺寸 > max_attn_size 的特征图，先降采样再计算注意力，
    然后将注意力图 upsample 回原始尺寸，避免 O(H^2 W^2) 显存爆炸。
    """

    def __init__(self, channels: int, num_heads: int = 4, max_attn_size: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_attn_size = max_attn_size

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query_feat.shape

        # 判断是否需要降采样
        need_downsample = H > self.max_attn_size or W > self.max_attn_size

        if need_downsample:
            # 计算降采样目标尺寸
            scale_h = min(self.max_attn_size / H, 1.0)
            scale_w = min(self.max_attn_size / W, 1.0)
            scale = min(scale_h, scale_w)
            new_h = max(int(H * scale), 4)
            new_w = max(int(W * scale), 4)

            q_input = F.interpolate(query_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            kv_input = F.interpolate(kv_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            q_input = query_feat
            kv_input = kv_feat
            new_h, new_w = H, W

        q = self.q_proj(q_input).view(B, self.num_heads, self.head_dim, new_h * new_w)
        k = self.k_proj(kv_input).view(B, self.num_heads, self.head_dim, new_h * new_w)
        v = self.v_proj(kv_input).view(B, self.num_heads, self.head_dim, new_h * new_w)

        # [B, heads, HW, HW] — 空间注意力
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, new_h, new_w)

        # 如果降采样了，需要 upsample 回原始尺寸
        if need_downsample:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return self.out_proj(out)


class SpatialGateFusion(nn.Module):
    """空间级门控融合：学习每个像素从哪个模态取多少信息。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(self, ir_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([ir_feat, vis_feat], dim=1)
        gates = torch.softmax(self.gate_net(combined), dim=1)
        return gates[:, 0:1] * ir_feat + gates[:, 1:2] * vis_feat


class CrossModalFusionBlock(nn.Module):
    """单层跨模态融合：通道注意力 + 空间门控 + 残差。"""

    def __init__(self, channels: int, num_heads: int = 4, max_attn_size: int = 32) -> None:
        super().__init__()
        self.ir_attend_vis = ChannelCrossAttention(channels, num_heads, max_attn_size)
        self.vis_attend_ir = ChannelCrossAttention(channels, num_heads, max_attn_size)
        self.spatial_gate = SpatialGateFusion(channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # 通道级跨模态交互
        ir_enhanced = ir + self.ir_attend_vis(ir, vis)
        vis_enhanced = vis + self.vis_attend_ir(vis, ir)

        # 空间门控选择
        fused = self.spatial_gate(ir_enhanced, vis_enhanced)

        # 残差精炼
        fused = fused + self.refine(fused)
        return self.act(fused)


class CrossModalQKVUnifiedFusion(nn.Module):
    """多尺度跨模态融合，P2-P5 各一个独立融合块（旧版，保留兼容）。

    F_fuse = f(F_ir, F_vis)

    对 P2/P3 大特征图使用降采样注意力 (max_attn_size=32)，
    对 P4/P5 小特征图使用完整注意力。
    """

    def __init__(self, channels: int = 128, num_scales: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        # P2, P3 用降采样注意力; P4, P5 用完整注意力
        attn_sizes = [32, 32, 64, 128]
        self.operators = nn.ModuleList(
            [
                CrossModalFusionBlock(channels, num_heads, max_attn_size=attn_sizes[i])
                for i in range(num_scales)
            ]
        )

    def forward(
        self,
        ir_features: tuple[torch.Tensor, ...],
        vis_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(ir_features) != len(vis_features) or len(ir_features) != len(self.operators):
            raise ValueError("CrossModalQKVUnifiedFusion expects aligned P2-P5 FPN scales.")
        return tuple(
            op(ir, vis)
            for op, ir, vis in zip(self.operators, ir_features, vis_features)
        )


# =============================================================================
# 新版：检测引导融合模块 (Task-Driven Detection-Guided Fusion)
# =============================================================================


class SaliencyConditionedChannelAttention(nn.Module):
    """显著性条件通道注意力：根据目标显著性图动态调整通道权重。

    目标区域显著性高时，增强与检测相关的语义通道；
    背景区域则保持纹理/细节通道。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        # saliency: [B, 1, H, W] -> 全局平均池化得到 [B, 1, 1, 1] 的显著性强度
        sal_strength = saliency.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1, 1]
        weight = self.channel_attn(sal_strength)  # [B, C]
        return feat * weight.view(weight.shape[0], weight.shape[1], 1, 1)


class DetectionGuidedFusionBlock(nn.Module):
    """单层检测引导跨模态融合。

    与旧版 CrossModalFusionBlock 的关键区别:
    1. 空间门控被 task saliency 显式调制 — 目标区域融合策略不同于背景
    2. 新增 saliency-conditioned channel attention: 目标区域增强跨模态交互
    3. 模态贡献权重由 TaskSaliencyPredictor 提供，而非纯自学习
    """

    def __init__(self, channels: int, num_heads: int = 4, max_attn_size: int = 32) -> None:
        super().__init__()
        # 双向跨模态注意力
        self.ir_attend_vis = ChannelCrossAttention(channels, num_heads, max_attn_size)
        self.vis_attend_ir = ChannelCrossAttention(channels, num_heads, max_attn_size)

        # 显著性条件通道注意力
        self.sal_channel_attn_ir = SaliencyConditionedChannelAttention(channels)
        self.sal_channel_attn_vis = SaliencyConditionedChannelAttention(channels)

        # 显著性条件空间门控 (输入多了 saliency 通道)
        self.saliency_gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )

        # 残差精炼
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        saliency: torch.Tensor,
        modal_weight: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 双向跨模态注意力交互
        ir_enhanced = ir + self.ir_attend_vis(ir, vis)
        vis_enhanced = vis + self.vis_attend_ir(vis, ir)

        # 2. 显著性条件通道注意力调制
        ir_enhanced = self.sal_channel_attn_ir(ir_enhanced, saliency)
        vis_enhanced = self.sal_channel_attn_vis(vis_enhanced, saliency)

        # 3. 显著性条件空间门控 + 模态权重联合决策
        gate_input = torch.cat([ir_enhanced, vis_enhanced, saliency], dim=1)
        gates = torch.softmax(self.saliency_gate(gate_input), dim=1)

        # 将学习到的门控与预测的模态权重相乘，实现双重引导
        ir_w = gates[:, 0:1] * modal_weight[:, 0:1]
        vis_w = gates[:, 1:2] * modal_weight[:, 1:2]
        norm = ir_w + vis_w + 1e-6
        fused = (ir_w / norm) * ir_enhanced + (vis_w / norm) * vis_enhanced

        # 4. 残差精炼
        fused = fused + self.refine(fused)
        return self.act(fused)


class DetectionGuidedFusion(nn.Module):
    """多尺度检测引导融合，替代原 CrossModalQKVUnifiedFusion。

    核心改进：融合过程在特征层面被检测任务显著性直接调制，
    而非仅通过损失权重间接影响。

    输入:
        ir_features: 红外 FPN 特征 (P2-P5)
        vis_features: 可见光 FPN 特征 (P2-P5)
        saliency_maps: 任务显著性图 (由 TaskSaliencyPredictor 生成)
        modal_weights: 模态贡献权重 (由 TaskSaliencyPredictor 生成)
    """

    def __init__(self, channels: int = 128, num_scales: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        attn_sizes = [32, 32, 64, 128]
        self.operators = nn.ModuleList(
            [
                DetectionGuidedFusionBlock(channels, num_heads, max_attn_size=attn_sizes[i])
                for i in range(num_scales)
            ]
        )

    def forward(
        self,
        ir_features: tuple[torch.Tensor, ...],
        vis_features: tuple[torch.Tensor, ...],
        saliency_maps: tuple[torch.Tensor, ...],
        modal_weights: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(ir_features) != len(vis_features) or len(ir_features) != len(self.operators):
            raise ValueError("DetectionGuidedFusion expects aligned P2-P5 FPN scales.")
        return tuple(
            op(ir, vis, sal, mw)
            for op, ir, vis, sal, mw in zip(
                self.operators, ir_features, vis_features, saliency_maps, modal_weights
            )
        )
