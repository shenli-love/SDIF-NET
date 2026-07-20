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


def charbonnier(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value.pow(2) + eps * eps)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class DetectionAwareFusionLoss(nn.Module):
    """Task-aware fusion objective for IR/VIS detection.

    Object boxes are supervised toward IR because thermal contrast is usually
    the detection cue. Background is supervised toward VIS so scene structure
    remains stable without letting bright VIS regions overwrite target texture.
    """

    def __init__(
        self,
        ssim_window: int = 7,
        object_weight: float = 2.0,
        background_weight: float = 1.0,
        gradient_weight: float = 1.0,
        ssim_weight: float = 0.5,
        detection_guided_weight: float = 0.5,
        modal_specific_weight: float = 0.5,
        missed_object_weight: float = 2.0,
        detected_object_weight: float = 0.5,
        detection_conf_threshold: float = 0.25,
        detection_iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.ssim_window = ssim_window
        self.object_weight = object_weight
        self.background_weight = background_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.detection_guided_weight = detection_guided_weight
        self.modal_specific_weight = modal_specific_weight
        self.missed_object_weight = missed_object_weight
        self.detected_object_weight = detected_object_weight
        self.detection_conf_threshold = detection_conf_threshold
        self.detection_iou_threshold = detection_iou_threshold

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        detections: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        object_map = self._build_weight_map(targets, fused.shape, fused.device, fused.dtype)
        background_map = 1.0 - object_map

        ir_grad = gradient_map(ir)
        vis_grad = gradient_map(vis)
        fused_grad = gradient_map(fused)

        object_intensity_loss = weighted_mean(charbonnier(fused - ir), object_map)
        background_intensity_loss = weighted_mean(charbonnier(fused - vis), background_map)
        object_gradient_loss = weighted_mean(charbonnier(fused_grad - ir_grad), object_map)
        background_gradient_loss = weighted_mean(
            charbonnier(fused_grad - vis_grad),
            background_map,
        )

        object_ssim_loss = weighted_mean(self._ssim_loss_map(fused, ir), object_map)
        background_ssim_loss = weighted_mean(self._ssim_loss_map(fused, vis), background_map)

        detection_attention = self._build_detection_attention_map(
            targets=targets,
            shape=fused.shape,
            device=fused.device,
            dtype=fused.dtype,
            detections=detections,
        )
        detection_guided_loss = self.detection_guided_weight * weighted_mean(
            charbonnier(fused - ir) + charbonnier(fused_grad - ir_grad),
            detection_attention,
        )
        modal_specific_loss = self.modal_specific_weight * self._modal_specific_loss(
            fused_grad=fused_grad,
            ir_grad=ir_grad,
            vis_grad=vis_grad,
        )

        object_loss = (
            object_intensity_loss
            + self.gradient_weight * object_gradient_loss
            + self.ssim_weight * object_ssim_loss
            + detection_guided_loss
        )
        background_loss = (
            background_intensity_loss
            + self.gradient_weight * background_gradient_loss
            + self.ssim_weight * background_ssim_loss
        )
        fusion_loss = (
            self.object_weight * object_loss
            + self.background_weight * background_loss
            + modal_specific_loss
        )
        zero = fused.new_tensor(0.0)
        return {
            "fusion_loss": fusion_loss,
            "object_intensity_loss": object_intensity_loss,
            "background_intensity_loss": background_intensity_loss,
            "object_gradient_loss": object_gradient_loss,
            "background_gradient_loss": background_gradient_loss,
            "object_ssim_loss": object_ssim_loss,
            "background_ssim_loss": background_ssim_loss,
            "object_loss": object_loss,
            "background_loss": background_loss,
            "detection_guided_loss": detection_guided_loss,
            "modal_specific_loss": modal_specific_loss,
            "detection_attention_mean": detection_attention.mean(),
            # Backward-compatible summaries for existing training logs.
            "intensity_loss": object_intensity_loss + background_intensity_loss,
            "gradient_loss": object_gradient_loss + background_gradient_loss,
            "ssim_loss": object_ssim_loss + background_ssim_loss,
            "nan_guard": zero,
        }

    def _modal_specific_loss(
        self,
        fused_grad: torch.Tensor,
        ir_grad: torch.Tensor,
        vis_grad: torch.Tensor,
    ) -> torch.Tensor:
        ir_unique = torch.sigmoid(8.0 * (ir_grad.detach() - vis_grad.detach()))
        vis_unique = torch.sigmoid(8.0 * (vis_grad.detach() - ir_grad.detach()))
        ir_term = weighted_mean(charbonnier(fused_grad - ir_grad), ir_unique)
        vis_term = weighted_mean(charbonnier(fused_grad - vis_grad), vis_unique)
        max_grad = torch.maximum(ir_grad, vis_grad)
        max_term = charbonnier(F.relu(max_grad.detach() - fused_grad)).mean()
        return ir_term + vis_term + max_term

    def _ssim_loss(self, fused: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._ssim_loss_map(fused, target).mean()

    def _ssim_loss_map(self, fused: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.ssim_window
        padding = window // 2
        mu_fused = F.avg_pool2d(
            fused,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        mu_ref = F.avg_pool2d(
            target,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        sigma_fused = F.avg_pool2d(
            fused * fused,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        ) - mu_fused.pow(2)
        sigma_ref = F.avg_pool2d(
            target * target,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        ) - mu_ref.pow(2)
        sigma_cross = F.avg_pool2d(
            fused * target,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        ) - mu_fused * mu_ref
        sigma_fused = sigma_fused.clamp_min(0.0)
        sigma_ref = sigma_ref.clamp_min(0.0)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (
            (2.0 * mu_fused * mu_ref + c1)
            * (2.0 * sigma_cross + c2)
        ) / (
            (mu_fused.pow(2) + mu_ref.pow(2) + c1)
            * (sigma_fused + sigma_ref + c2)
        ).clamp_min(1e-6)
        return torch.nan_to_num(1.0 - ssim.clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def _build_weight_map(
        targets: list[dict[str, torch.Tensor]],
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, _, height, width = shape
        mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        for batch_idx, target in enumerate(targets):
            if batch_idx >= batch_size:
                break
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                continue
            if target.get("box_format", "cxcywh") == "cxcywh":
                boxes = cxcywh_to_xyxy(boxes)
            boxes = boxes.clamp(0.0, 1.0)
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                left = max(0, min(width - 1, int(torch.floor(box.new_tensor(x1 * width)).item())))
                top = max(0, min(height - 1, int(torch.floor(box.new_tensor(y1 * height)).item())))
                right = max(left + 1, min(width, int(torch.ceil(box.new_tensor(x2 * width)).item())))
                bottom = max(top + 1, min(height, int(torch.ceil(box.new_tensor(y2 * height)).item())))
                mask[batch_idx, :, top:bottom, left:right] = 1.0
        return mask

    def _build_detection_attention_map(
        self,
        targets: list[dict[str, torch.Tensor]],
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        detections: dict[str, object] | None,
    ) -> torch.Tensor:
        batch_size, _, height, width = shape
        attention = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        if detections is None:
            return attention

        decoded = detections.get("decoded", detections)
        if not isinstance(decoded, dict):
            return attention
        if "boxes" not in decoded or "scores" not in decoded:
            return attention

        pred_boxes = decoded["boxes"]
        pred_scores = decoded["scores"]
        if not torch.is_tensor(pred_boxes) or not torch.is_tensor(pred_scores):
            return attention

        pred_boxes = pred_boxes.detach().to(device=device, dtype=torch.float32)
        pred_scores = pred_scores.detach().to(device=device, dtype=torch.float32)
        if "class_logits" in decoded and torch.is_tensor(decoded["class_logits"]):
            class_prob = F.softmax(
                decoded["class_logits"].detach().to(device=device, dtype=torch.float32),
                dim=-1,
            ).amax(dim=-1)
            pred_scores = pred_scores * class_prob
        pred_scores = pred_scores.clamp(0.0, 1.0)

        for batch_idx, target in enumerate(targets):
            if batch_idx >= batch_size:
                break
            gt_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            if gt_boxes.numel() == 0:
                continue
            if target.get("box_format", "cxcywh") == "cxcywh":
                gt_boxes = cxcywh_to_xyxy(gt_boxes)
            gt_boxes = gt_boxes.clamp(0.0, 1.0)

            cur_boxes = pred_boxes[batch_idx].clamp(0.0, 1.0)
            cur_scores = pred_scores[batch_idx]
            if cur_boxes.numel() == 0:
                quality = torch.zeros(gt_boxes.shape[0], device=device, dtype=torch.float32)
            else:
                ious = box_iou(gt_boxes, cur_boxes)
                best_iou, best_idx = ious.max(dim=1)
                matched_score = cur_scores[best_idx]
                matched_score = torch.where(
                    matched_score >= self.detection_conf_threshold,
                    matched_score,
                    torch.zeros_like(matched_score),
                )
                quality = torch.where(
                    best_iou >= self.detection_iou_threshold,
                    matched_score,
                    torch.zeros_like(matched_score),
                )

            for box, score in zip(gt_boxes, quality):
                # Missed or low-confidence objects get a stronger IR-alignment
                # penalty, making detection feedback alter the fusion image.
                hardness = (1.0 - score).clamp(0.0, 1.0)
                value = self.detected_object_weight + (
                    self.missed_object_weight - self.detected_object_weight
                ) * hardness
                x1, y1, x2, y2 = box.tolist()
                left = max(0, min(width - 1, int(torch.floor(box.new_tensor(x1 * width)).item())))
                top = max(0, min(height - 1, int(torch.floor(box.new_tensor(y1 * height)).item())))
                right = max(left + 1, min(width, int(torch.ceil(box.new_tensor(x2 * width)).item())))
                bottom = max(top + 1, min(height, int(torch.ceil(box.new_tensor(y2 * height)).item())))
                attention[batch_idx, :, top:bottom, left:right] = torch.maximum(
                    attention[batch_idx, :, top:bottom, left:right],
                    value.to(device=device, dtype=dtype),
                )

        return attention

FusionReconstructionLoss = DetectionAwareFusionLoss


class YOLOLikeDetectionLoss(nn.Module):
    """Simple one-positive-per-GT loss for the placeholder dense detector."""

    def __init__(
        self,
        num_classes: int,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        noobj_weight: float = 0.1,
        small_object_boost: float = 1.0,
        small_object_area_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.noobj_weight = noobj_weight
        self.small_object_boost = small_object_boost
        self.small_object_area_threshold = small_object_area_threshold

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
            target_cxcywh = xyxy_to_cxcywh(gt_xyxy)
            pred_cxcywh = xyxy_to_cxcywh(pred_pos_boxes)
            gt_area = (target_cxcywh[:, 2] * target_cxcywh[:, 3]).clamp_min(0.0)
            small_weight = 1.0 + self.small_object_boost * (
                self.small_object_area_threshold - gt_area
            ).clamp_min(0.0) / self.small_object_area_threshold
            total_box = total_box + ((1.0 - best_iou) * small_weight).mean()
            l1_per_box = F.l1_loss(
                pred_cxcywh,
                target_cxcywh,
                reduction="none",
            ).mean(dim=1)
            total_box = total_box + (l1_per_box * small_weight).mean()
            obj_loss = F.binary_cross_entropy(
                scores_pred[batch_idx, best_idx].clamp(1e-4, 1.0 - 1e-4),
                torch.ones_like(best_iou),
                reduction="none",
            )
            total_obj = total_obj + (obj_loss * small_weight).mean()
            cls_loss = F.cross_entropy(
                class_logits[batch_idx, best_idx],
                gt_labels.clamp(min=0, max=self.num_classes - 1),
                reduction="none",
            )
            total_cls = total_cls + (cls_loss * small_weight).mean()
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
        object_weight: float = 2.0,
        background_weight: float = 1.0,
        gradient_weight: float = 1.0,
        ssim_weight: float = 0.5,
        modal_specific_weight: float = 0.5,
        small_object_boost: float = 1.0,
    ) -> None:
        super().__init__()
        self.fusion_weight = fusion_weight
        self.detection_weight = detection_weight
        self.use_feedback = use_feedback
        self.fusion_loss = DetectionAwareFusionLoss(
            object_weight=object_weight,
            background_weight=background_weight,
            gradient_weight=gradient_weight,
            ssim_weight=ssim_weight,
            modal_specific_weight=modal_specific_weight,
        )
        self.detection_loss = YOLOLikeDetectionLoss(
            num_classes=num_classes,
            small_object_boost=small_object_boost,
        )
        self.feedback_weight = DetectionFeedbackLossWeight(base_lambda=detection_weight)

    def forward(
        self,
        outputs: dict[str, object],
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        use_feedback: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        detections = outputs["detections"]
        fusion_terms = self.fusion_loss(
            outputs["I_fused"],
            ir,
            vis,
            targets,
            detections=detections,
        )
        decoded = detections["decoded"]
        if detections.get("trainable", True):
            detection_terms = self.detection_loss(decoded, targets)
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
