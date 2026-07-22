from __future__ import annotations

import torch
from torch import nn

from ..utils.boxes import box_iou, cxcywh_to_xyxy


class DetectionFeedbackLossWeight(nn.Module):
    """基于 F1 + 定位质量的动态损失权重。

    解决旧版问题：密集锚框导致 recall 永远=1。
    新方案：综合考虑 precision、定位精度、分类置信度。

        L_total = L_fusion + lambda_det * L_detection

    lambda_det grows when F1 or localization quality is low.
    """

    def __init__(
        self,
        base_lambda: float = 1.0,
        min_lambda: float = 0.5,
        max_lambda: float = 3.0,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_lambda = base_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(
        self,
        detections: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        boxes = detections["boxes"].detach()
        scores = detections["scores"].detach()
        device = boxes.device
        dtype = boxes.dtype

        recalls, precisions, loc_qualities = [], [], []

        for batch_idx, target in enumerate(targets):
            gt_boxes = self._target_boxes_xyxy(target, device)
            pred_scores = scores[batch_idx]
            keep = pred_scores >= self.score_threshold
            pred_boxes = boxes[batch_idx][keep]
            kept_scores = pred_scores[keep]

            if gt_boxes.numel() == 0:
                recalls.append(1.0)
                precisions.append(1.0 if pred_boxes.numel() == 0 else 0.0)
                loc_qualities.append(1.0)
                continue
            if pred_boxes.numel() == 0:
                recalls.append(0.0)
                precisions.append(0.0)
                loc_qualities.append(0.0)
                continue

            ious = box_iou(gt_boxes, pred_boxes)
            best_iou, best_idx = ious.max(dim=1)
            matched = best_iou >= self.iou_threshold

            # Recall: GT 被匹配的比例
            recall_val = float(matched.float().mean().item())
            recalls.append(recall_val)

            # Precision: 预测框中有多少是 TP (关键！密集锚框下这个很低)
            # 对每个预测框，检查它与最佳GT的IoU
            pred_best_iou = ious.max(dim=0).values
            pred_matched = pred_best_iou >= self.iou_threshold
            precision_val = float(pred_matched.float().mean().item())
            precisions.append(precision_val)

            # 定位质量：匹配框的平均 IoU
            if matched.any():
                loc_qualities.append(float(best_iou[matched].mean().item()))
            else:
                loc_qualities.append(0.0)

        recall = torch.tensor(
            sum(recalls) / max(len(recalls), 1), device=device, dtype=dtype
        )
        precision = torch.tensor(
            sum(precisions) / max(len(precisions), 1), device=device, dtype=dtype
        )
        loc_quality = torch.tensor(
            sum(loc_qualities) / max(len(loc_qualities), 1), device=device, dtype=dtype
        )

        # F1 综合指标
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

        # 综合难度：F1低 + 定位差 → hardness高
        quality_score = 0.5 * f1 + 0.3 * loc_quality + 0.2 * recall
        hardness = (1.0 - quality_score).clamp(0.0, 1.0)

        if enabled:
            lambda_det = self.base_lambda * (1.0 + 2.0 * hardness)
        else:
            lambda_det = torch.tensor(self.base_lambda, device=device, dtype=dtype)

        lambda_det = lambda_det.clamp(self.min_lambda, self.max_lambda)

        # 对外报告 recall 和 confidence (兼容日志)
        confidence = loc_quality
        return {
            "lambda_det": lambda_det,
            "detection_recall": recall,
            "detection_confidence": confidence,
        }

    @staticmethod
    def _target_boxes_xyxy(
        target: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        if boxes.numel() == 0:
            return torch.zeros((0, 4), device=device)
        if target.get("box_format", "cxcywh") == "cxcywh":
            boxes = cxcywh_to_xyxy(boxes)
        return boxes.clamp(0.0, 1.0)
