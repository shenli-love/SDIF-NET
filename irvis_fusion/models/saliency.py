from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TaskSaliencyPredictor(nn.Module):
    """从双模态 FPN 特征中预测任务显著性图和模态贡献权重。

    核心作用：为融合模块提供空间级检测任务先验，
    让融合过程知道"哪里有目标"以及"该区域该信任哪个模态"。

    输出:
        saliency_maps: tuple of [B, 1, H_i, W_i] 每个尺度的目标显著性 (0~1)
        modal_weights: tuple of [B, 2, H_i, W_i] 每个尺度 IR/VIS 的信息贡献权重 (softmax)
    """

    def __init__(self, channels: int = 128, num_scales: int = 4) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.saliency_heads = nn.ModuleList()
        self.modal_heads = nn.ModuleList()
        for _ in range(num_scales):
            self.saliency_heads.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2, channels, 3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels // 2, 3, padding=1),
                    nn.GroupNorm(8, channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels // 2, 1, 1),
                    nn.Sigmoid(),
                )
            )
            self.modal_heads.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2, channels, 3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, 2, 1),
                )
            )

    def forward(
        self,
        ir_features: tuple[torch.Tensor, ...],
        vis_features: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        saliency_maps = []
        modal_weights = []
        for i in range(self.num_scales):
            combined = torch.cat([ir_features[i], vis_features[i]], dim=1)
            saliency_maps.append(self.saliency_heads[i](combined))
            modal_weights.append(torch.softmax(self.modal_heads[i](combined), dim=1))
        return tuple(saliency_maps), tuple(modal_weights)
