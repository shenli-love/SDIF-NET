from __future__ import annotations

import torch
from torch import nn

from ..utils.boxes import box_iou, cxcywh_to_xyxy, draw_normalized_boxes


class DetectionFeedback(nn.Module):
    """Create miss and uncertainty masks from detector predictions and GT."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.25,
        uncertainty_topk: int = 200,
    ) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.uncertainty_topk = uncertainty_topk

    @torch.no_grad()
    def forward(
        self,
        detections: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor | float]:
        boxes = detections["boxes"].detach()
        scores = detections["scores"].detach()
        height, width = image_size
        batch_size = boxes.shape[0]
        miss_masks = []
        uncertainty_masks = []
        recalls = []
        mean_confidences = []
        for batch_idx in range(batch_size):
            cur_boxes = boxes[batch_idx]
            cur_scores = scores[batch_idx]
            score_keep = cur_scores >= self.score_threshold
            kept_boxes = cur_boxes[score_keep]
            kept_scores = cur_scores[score_keep]
            target = targets[batch_idx] if targets is not None else None
            gt_boxes = self._target_boxes_xyxy(target, device=boxes.device)
            miss_boxes = gt_boxes
            matched_scores = []
            if gt_boxes.numel() > 0 and kept_boxes.numel() > 0:
                ious = box_iou(gt_boxes, kept_boxes)
                best_iou, best_idx = ious.max(dim=1)
                matched = best_iou >= self.iou_threshold
                miss_boxes = gt_boxes[~matched]
                if matched.any():
                    matched_scores = kept_scores[best_idx[matched]].tolist()
                recalls.append(float(matched.float().mean().item()))
            elif gt_boxes.numel() > 0:
                recalls.append(0.0)
            else:
                recalls.append(1.0)

            if len(matched_scores) > 0:
                mean_confidences.append(float(sum(matched_scores) / len(matched_scores)))
            elif kept_scores.numel() > 0:
                mean_confidences.append(float(kept_scores.mean().item()))
            else:
                mean_confidences.append(0.0)

            miss_masks.append(
                draw_normalized_boxes(
                    miss_boxes,
                    1.0,
                    height,
                    width,
                    device=boxes.device,
                    dtype=boxes.dtype,
                )
            )

            uncertainty_masks.append(
                self._uncertainty_mask(cur_boxes, cur_scores, height, width)
            )

        miss = torch.stack(miss_masks, dim=0)
        uncertainty = torch.stack(uncertainty_masks, dim=0)
        feedback = (miss + uncertainty).clamp(0.0, 1.0)
        return {
            "M_miss": miss,
            "U": uncertainty,
            "G_fb": feedback,
            "recall": float(sum(recalls) / max(len(recalls), 1)),
            "mean_confidence": float(
                sum(mean_confidences) / max(len(mean_confidences), 1)
            ),
        }

    def _uncertainty_mask(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if scores.numel() == 0:
            return torch.zeros((1, height, width), device=boxes.device, dtype=boxes.dtype)
        k = min(self.uncertainty_topk, scores.numel())
        top_scores, top_idx = torch.topk(scores, k=k)
        uncertain_boxes = boxes[top_idx]
        uncertainty = 1.0 - top_scores
        return draw_normalized_boxes(
            uncertain_boxes,
            uncertainty,
            height,
            width,
            device=boxes.device,
            dtype=boxes.dtype,
        )

    @staticmethod
    def _target_boxes_xyxy(
        target: dict[str, torch.Tensor] | None,
        device: torch.device,
    ) -> torch.Tensor:
        if target is None or "boxes" not in target or target["boxes"].numel() == 0:
            return torch.zeros((0, 4), device=device)
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        fmt = target.get("box_format", "cxcywh")
        if fmt == "cxcywh":
            boxes = cxcywh_to_xyxy(boxes)
        elif fmt != "xyxy":
            raise ValueError(f"Unsupported target box format: {fmt}")
        return boxes.clamp(0.0, 1.0)
