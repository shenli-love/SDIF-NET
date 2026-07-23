from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DetectionFeedbackModulator(nn.Module):
    """将检测头的中间特征回传到融合层，实现特征级闭环反馈。

    核心思想：检测头在预测过程中产生的中间特征包含了"当前融合特征
    对检测任务是否充分"的信息。通过将这些信息回传并调制融合特征，
    形成训练时的闭环，让融合模块直接感知检测任务的需求。

    训练时: detector 的中间特征 → 生成 modulation signal → 调制 fused features
    推理时: 禁用此回路（单 pass，无反馈）

    调制方式: feat_modulated = feat * (1 + sigmoid(modulator(det_feat)))
    这确保调制范围在 [1, 2]，不会破坏原始特征，只做增强。
    """

    def __init__(
        self,
        det_channels: int = 128,
        fusion_channels: int = 128,
        num_scales: int = 4,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.modulators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(det_channels, fusion_channels, 1),
                    nn.GroupNorm(8, fusion_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
                    nn.GroupNorm(8, fusion_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fusion_channels, fusion_channels, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ]
        )

    def forward(
        self,
        fused_features: tuple[torch.Tensor, ...],
        det_intermediate: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """用检测中间特征调制融合特征。

        Args:
            fused_features: 融合模块输出的 P2-P5 特征
            det_intermediate: 检测头各层的中间特征 (与 fused_features 对齐)

        Returns:
            调制后的融合特征
        """
        modulated = []
        for feat, det_feat, mod in zip(fused_features, det_intermediate, self.modulators):
            # 确保空间尺寸对齐
            if det_feat.shape[-2:] != feat.shape[-2:]:
                det_feat = F.interpolate(
                    det_feat, size=feat.shape[-2:], mode="bilinear", align_corners=False
                )
            # 确保通道数对齐
            if det_feat.shape[1] != feat.shape[1]:
                det_feat = det_feat[:, : feat.shape[1]]
            modulation = mod(det_feat)
            modulated.append(feat * (1.0 + modulation))
        return tuple(modulated)
