from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..models.feedback import DetectionFeedbackLossWeight
from .boxes import box_iou, cxcywh_to_xyxy, xyxy_to_cxcywh


def gradient_map(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx + dy


class FusionReconstructionLoss(nn.Module):
    """Intensity and gradient losses for infrared-visible fusion."""

    def __init__(self, intensity_weight: float = 1.0, gradient_weight: float = 5.0) -> None:
        super().__init__()
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_intensity = torch.maximum(ir, vis)
        intensity = F.l1_loss(fused, target_intensity)
        target_grad = torch.maximum(gradient_map(ir), gradient_map(vis))
        gradient = F.l1_loss(gradient_map(fused), target_grad)
        sam_consistency = fused.new_tensor(0.0)
        if sam_mask is not None:
            sam = sam_mask.float().clamp(0.0, 1.0)
            sam_consistency = F.l1_loss(fused * sam, target_intensity * sam)
        loss = (
            self.intensity_weight * intensity
            + self.gradient_weight * gradient
            + sam_consistency
        )
        return {
            "fusion_loss": loss,
            "intensity_loss": intensity,
            "gradient_loss": gradient,
            "sam_consistency_loss": sam_consistency,
        }


class YOLOLikeDetectionLoss(nn.Module):
    """Simple one-positive-per-GT loss for the placeholder dense detector."""

    def __init__(
        self,
        num_classes: int,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        noobj_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.noobj_weight = noobj_weight

    def forward(
        self,
        detections: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        boxes_pred = detections["boxes"]
        scores_pred = detections["scores"]
        class_logits = detections["class_logits"]
        device = boxes_pred.device
        total_box = boxes_pred.new_tensor(0.0)
        total_obj = boxes_pred.new_tensor(0.0)
        total_cls = boxes_pred.new_tensor(0.0)
        total_noobj = boxes_pred.new_tensor(0.0)
        pos_count = 0
        batch_size = boxes_pred.shape[0]

        for batch_idx, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            gt_labels = target["labels"].to(device=device, dtype=torch.long)
            if target.get("box_format", "cxcywh") == "cxcywh":
                gt_xyxy = cxcywh_to_xyxy(gt_boxes)
            else:
                gt_xyxy = gt_boxes
            gt_xyxy = gt_xyxy.clamp(0.0, 1.0)
            if gt_xyxy.numel() == 0:
                total_noobj = total_noobj + scores_pred[batch_idx].pow(2).mean()
                continue

            ious = box_iou(gt_xyxy, boxes_pred[batch_idx])
            best_iou, best_idx = ious.max(dim=1)
            pred_pos_boxes = boxes_pred[batch_idx, best_idx]
            total_box = total_box + (1.0 - best_iou).mean()
            total_box = total_box + F.l1_loss(
                xyxy_to_cxcywh(pred_pos_boxes),
                xyxy_to_cxcywh(gt_xyxy),
            )
            total_obj = total_obj + F.binary_cross_entropy(
                scores_pred[batch_idx, best_idx].clamp(1e-4, 1.0 - 1e-4),
                torch.ones_like(best_iou),
            )
            total_cls = total_cls + F.cross_entropy(
                class_logits[batch_idx, best_idx],
                gt_labels.clamp(min=0, max=self.num_classes - 1),
            )
            pos_count += gt_xyxy.shape[0]

            neg_mask = torch.ones(
                scores_pred.shape[1],
                dtype=torch.bool,
                device=device,
            )
            neg_mask[best_idx] = False
            if neg_mask.any():
                total_noobj = total_noobj + scores_pred[batch_idx, neg_mask].pow(2).mean()

        norm = max(batch_size, 1)
        box_loss = total_box / norm
        obj_loss = total_obj / norm
        cls_loss = total_cls / norm
        noobj_loss = total_noobj / norm
        loss = (
            self.box_weight * box_loss
            + self.obj_weight * obj_loss
            + self.cls_weight * cls_loss
            + self.noobj_weight * noobj_loss
        )
        return {
            "detection_loss": loss,
            "box_loss": box_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
            "noobj_loss": noobj_loss,
            "positive_count": boxes_pred.new_tensor(float(pos_count)),
        }


class JointFusionDetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        fusion_weight: float = 1.0,
        detection_weight: float = 1.0,
        use_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.fusion_weight = fusion_weight
        self.detection_weight = detection_weight
        self.use_feedback = use_feedback
        self.fusion_loss = FusionReconstructionLoss()
        self.detection_loss = YOLOLikeDetectionLoss(num_classes=num_classes)
        self.feedback_weight = DetectionFeedbackLossWeight(base_lambda=detection_weight)

    def forward(
        self,
        outputs: dict[str, object],
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_mask: torch.Tensor | None,
        targets: list[dict[str, torch.Tensor]],
        use_feedback: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        fusion_terms = self.fusion_loss(outputs["I_fused"], ir, vis, sam_mask)
        decoded = outputs["detections"]["decoded"]
        if outputs["detections"].get("trainable", True):
            detection_terms = self.detection_loss(
                decoded,
                targets,
            )
        else:
            zero = outputs["I_fused"].new_tensor(0.0)
            detection_terms = {
                "detection_loss": zero,
                "box_loss": zero,
                "obj_loss": zero,
                "cls_loss": zero,
                "noobj_loss": zero,
                "positive_count": zero,
            }
        enabled = self.use_feedback if use_feedback is None else use_feedback
        feedback_terms = self.feedback_weight(
            decoded,
            targets,
            enabled=enabled,
        )
        lambda_det = feedback_terms["lambda_det"]
        total = (
            self.fusion_weight * fusion_terms["fusion_loss"]
            + lambda_det * detection_terms["detection_loss"]
        )
        return {
            "loss": total,
            **fusion_terms,
            **detection_terms,
            **feedback_terms,
        }
