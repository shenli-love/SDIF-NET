from __future__ import annotations

import torch
from torch import nn

from ..utils.boxes import box_iou, cxcywh_to_xyxy


class DetectionFeedbackLossWeight(nn.Module):
    """Convert detection quality into a dynamic loss weight.

    This replaces the previous forward-time mask feedback. Detection feedback
    now influences only the training objective:

        L_total = L_fusion + lambda_det * L_detection

    lambda_det grows when confidence or recall is low.
    """

    def __init__(
        self,
        base_lambda: float = 1.0,
        min_lambda: float = 0.25,
        max_lambda: float = 2.5,
        score_threshold: float = 0.25,
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
        recalls = []
        confidences = []
        for batch_idx, target in enumerate(targets):
            gt_boxes = self._target_boxes_xyxy(target, device)
            pred_scores = scores[batch_idx]
            keep = pred_scores >= self.score_threshold
            pred_boxes = boxes[batch_idx][keep]
            kept_scores = pred_scores[keep]
            if gt_boxes.numel() == 0:
                recalls.append(1.0)
                confidences.append(float(kept_scores.mean().item()) if kept_scores.numel() else 1.0)
                continue
            if pred_boxes.numel() == 0:
                recalls.append(0.0)
                confidences.append(0.0)
                continue
            ious = box_iou(gt_boxes, pred_boxes)
            best_iou, best_idx = ious.max(dim=1)
            matched = best_iou >= self.iou_threshold
            recalls.append(float(matched.float().mean().item()))
            if matched.any():
                confidences.append(float(kept_scores[best_idx[matched]].mean().item()))
            else:
                confidences.append(float(kept_scores.mean().item()) if kept_scores.numel() else 0.0)

        recall = torch.tensor(
            sum(recalls) / max(len(recalls), 1),
            device=device,
            dtype=dtype,
        )
        confidence = torch.tensor(
            sum(confidences) / max(len(confidences), 1),
            device=device,
            dtype=dtype,
        )
        if enabled:
            hardness = 0.5 * (1.0 - recall) + 0.5 * (1.0 - confidence)
            lambda_det = self.base_lambda * (1.0 + hardness)
        else:
            lambda_det = torch.tensor(self.base_lambda, device=device, dtype=dtype)
        lambda_det = lambda_det.clamp(self.min_lambda, self.max_lambda)
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
