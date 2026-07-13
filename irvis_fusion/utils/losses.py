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


def laplacian_map(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return torch.abs(F.conv2d(x, kernel, padding=1, groups=x.shape[1]))


def box_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def local_contrast_map(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    return torch.abs(x - box_blur(x, kernel_size))


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pad = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x_sq
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / denominator.clamp_min(1e-6)
    return 1.0 - ssim.clamp(0.0, 1.0).mean()


def multiscale_structure_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    loss = x.new_tensor(0.0)
    valid_scales = 0
    for scale in scales:
        if scale > 1:
            if min(x.shape[-2:]) < scale:
                continue
            cur_x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            cur_y = F.avg_pool2d(y, kernel_size=scale, stride=scale)
        else:
            cur_x = x
            cur_y = y
        loss = loss + F.l1_loss(gradient_map(cur_x), gradient_map(cur_y))
        loss = loss + 0.5 * F.l1_loss(laplacian_map(cur_x), laplacian_map(cur_y))
        valid_scales += 1
    return loss / max(valid_scales, 1)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class DetectionAwareFusionLoss(nn.Module):
    """Fusion loss with GT-box spatial weighting for detection-aware training."""

    def __init__(
        self,
        intensity_weight: float = 0.1,
        gradient_weight: float = 1.0,
        object_region_weight: float = 0.5,
        sam_weight: float = 1.0,
        ssim_weight: float = 0.5,
        perceptual_weight: float = 0.2,
        # target_max_weight: float = 1,
        excess_weight: float = 0.35,
        flat_smoothness_weight: float = 0.02,
        ring_artifact_weight: float = 0.25,
        thermal_preserve_weight: float = 1.2,
        halo_artifact_weight: float = 0.25,
        texture_blur_kernel: int = 5,
        contrast_blur_kernel: int = 9,
        thermal_margin: float = 0.03,
        thermal_temperature: float = 18.0,
        excess_margin: float = 0.03,
        gradient_overshoot_margin: float = 0.01,
        flat_gradient_threshold: float = 0.01,
        ring_width: int = 15,
    ) -> None:
        super().__init__()
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
        self.object_region_weight = object_region_weight
        self.sam_weight = sam_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        # self.target_max_weight = target_max_weight
        self.excess_weight = excess_weight
        self.flat_smoothness_weight = flat_smoothness_weight
        self.ring_artifact_weight = ring_artifact_weight
        self.thermal_preserve_weight = thermal_preserve_weight
        self.halo_artifact_weight = halo_artifact_weight
        self.texture_blur_kernel = texture_blur_kernel
        self.contrast_blur_kernel = contrast_blur_kernel
        self.thermal_margin = thermal_margin
        self.thermal_temperature = thermal_temperature
        self.excess_margin = excess_margin
        self.gradient_overshoot_margin = gradient_overshoot_margin
        self.flat_gradient_threshold = flat_gradient_threshold
        self.ring_width = ring_width

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        sam_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # ================= 1. 计算纹理能量 =================
        # 先计算梯度图代表纹理能量
        ir_grad = gradient_map(ir)
        vis_grad = gradient_map(vis)
        object_weight_map = self._build_weight_map(targets, fused.shape, fused.device, fused.dtype)
        if sam_mask is not None:
            sam = sam_mask.float().clamp(0.0, 1.0)
        else:
            sam = None

        # Use a hard source target again, but make the decision from smoothed
        # structure energy rather than raw per-pixel gradients. This keeps the
        # old crispness without the salt-and-pepper supervision.
        ir_structure = box_blur(
            ir_grad + 0.5 * local_contrast_map(ir, self.contrast_blur_kernel),
            self.texture_blur_kernel,
        )
        vis_structure = box_blur(
            vis_grad + 0.5 * local_contrast_map(vis, self.contrast_blur_kernel),
            self.texture_blur_kernel,
        )
        texture_mask = (ir_structure >= vis_structure).to(dtype=fused.dtype)

        thermal_score = torch.sigmoid(
            (ir - vis - self.thermal_margin) * self.thermal_temperature
        )
        focus_mask = object_weight_map
        if sam is not None:
            focus_mask = torch.maximum(focus_mask, sam.to(dtype=fused.dtype))
        focused_thermal_mask = torch.sigmoid((ir - vis) * self.thermal_temperature)
        thermal_score = torch.maximum(thermal_score, focus_mask * focused_thermal_mask)
        thermal_mask = (thermal_score > 0.5).to(dtype=fused.dtype)

        # ================= 2. 构建“纹理优势参考图” =================
        texture_target = torch.where(texture_mask > 0.5, ir, vis)
        target_struct_img = torch.where(thermal_mask > 0.5, ir, texture_target)

        # 保留 max_intensity 仅用于抑制过曝，不做训练目标！
        max_intensity = torch.maximum(ir, vis)

        # ================= 3. 结构损失和感知损失 =================
        # 直接对比融合图与“纹理优势参考图”，彻底代替之前死板的 ir
        target_grad = gradient_map(target_struct_img)
        max_grad = torch.maximum(ir_grad, vis_grad)
        grad_weight = (box_blur(target_grad, self.texture_blur_kernel) > self.flat_gradient_threshold).to(
            dtype=fused.dtype
        )
        grad_weight = torch.maximum(grad_weight, object_weight_map)
        if sam is not None:
            grad_weight = torch.maximum(grad_weight, sam.to(dtype=fused.dtype))
        fused_grad = gradient_map(fused)
        gradient = weighted_mean((fused_grad - target_grad).abs(), grad_weight)
        structure = ssim_loss(fused, target_struct_img)
        perceptual = multiscale_structure_loss(fused, target_struct_img)

        intensity = F.l1_loss(fused, target_struct_img)
        thermal_preserve = weighted_mean(F.relu(ir - fused), thermal_score)

        # ================= 4. 目标区域加权损失 =================
        abs_error = (fused - target_struct_img).abs()  # 目标区域比对纹理优势图
        object_region = fused.new_tensor(0.0)
        if object_weight_map.sum() > 0:
            object_region = (abs_error * object_weight_map).sum() / object_weight_map.sum().clamp_min(1.0)

        # ================= 5. 过曝抑制（保持不变）=================
        excess = F.relu(fused - max_intensity - self.excess_margin)
        if object_weight_map.sum() > 0:
            excess_weight_map = torch.clamp(object_weight_map + 0.25, max=1.0)
            excess_artifact = weighted_mean(excess, excess_weight_map)
        else:
            excess_artifact = excess.mean()

        edge_mask = (
            box_blur(torch.maximum(target_grad, max_grad), self.texture_blur_kernel)
            > self.flat_gradient_threshold
        ).to(dtype=fused.dtype)
        halo_artifact = weighted_mean(
            F.relu(fused - max_intensity - self.excess_margin)
            + 0.25 * F.relu(fused_grad - target_grad - self.gradient_overshoot_margin),
            edge_mask,
        )

        # ================= 6. 平坦平滑损失（保持不变）=================
        flat_mask = (torch.maximum(ir_grad, vis_grad) < self.flat_gradient_threshold).to(dtype=fused.dtype)
        flat_smoothness = weighted_mean(gradient_map(fused), flat_mask)

        # ================= 7. SAM 相关（保持纹理优先）=================
        sam_consistency = fused.new_tensor(0.0)
        ring_artifact = fused.new_tensor(0.0)
        if sam is not None:
            # SAM 掩膜区域的纹理，同样和“纹理优势参考图”对比
            sam_consistency = F.l1_loss(fused * sam, target_struct_img * sam)
            
            ring_mask = self._build_sam_ring(sam, self.ring_width)
            if ring_mask.sum() > 0:
                ring_excess = F.relu(fused - ir - self.excess_margin)
                ring_artifact = weighted_mean(ring_excess, ring_mask)
                ring_artifact = ring_artifact + 0.25 * weighted_mean(
                    gradient_map(fused),
                    ring_mask,
                )

        # ================= 8. 总 Loss 求和 =================
        loss = (
            self.intensity_weight * intensity
            + self.gradient_weight * gradient
            + self.object_region_weight * object_region
            + self.sam_weight * sam_consistency
            + self.ssim_weight * structure
            + self.perceptual_weight * perceptual
            + self.excess_weight * excess_artifact
            + self.flat_smoothness_weight * flat_smoothness
            + self.ring_artifact_weight * ring_artifact
            + self.thermal_preserve_weight * thermal_preserve
            + self.halo_artifact_weight * halo_artifact
        )
        return {
            "fusion_loss": loss,
            "gradient_loss": gradient,
            "object_region_loss": object_region,
            "sam_consistency_loss": sam_consistency,
            "ssim_loss": structure,
            "perceptual_loss": perceptual,
            "excess_artifact_loss": excess_artifact,
            "flat_smoothness_loss": flat_smoothness,
            "ring_artifact_loss": ring_artifact,
            "thermal_preserve_loss": thermal_preserve,
            "halo_artifact_loss": halo_artifact,
            "object_region_pixels": object_weight_map.sum(),
        }

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

    @staticmethod
    def _build_sam_ring(sam_mask: torch.Tensor, ring_width: int) -> torch.Tensor:
        if ring_width <= 0:
            return torch.zeros_like(sam_mask)
        binary = (sam_mask > 0.5).to(dtype=sam_mask.dtype)
        if binary.sum() == 0:
            return torch.zeros_like(binary)
        kernel_size = ring_width * 2 + 1
        dilated = F.max_pool2d(
            binary,
            kernel_size=kernel_size,
            stride=1,
            padding=ring_width,
        )
        return (dilated - binary).clamp(0.0, 1.0)


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
        # target_max_weight: float = 0.8,
        excess_weight: float = 0.25,
        flat_smoothness_weight: float = 0.02,
        ring_artifact_weight: float = 0.25,
        thermal_preserve_weight: float = 1.2,
        halo_artifact_weight: float = 0.25,
        texture_blur_kernel: int = 5,
        contrast_blur_kernel: int = 9,
        thermal_margin: float = 0.03,
        thermal_temperature: float = 18.0,
        gradient_overshoot_margin: float = 0.01,
    ) -> None:
        super().__init__()
        self.fusion_weight = fusion_weight
        self.detection_weight = detection_weight
        self.use_feedback = use_feedback
        self.fusion_loss = DetectionAwareFusionLoss(
            excess_weight=excess_weight,
            flat_smoothness_weight=flat_smoothness_weight,
            ring_artifact_weight=ring_artifact_weight,
            thermal_preserve_weight=thermal_preserve_weight,
            halo_artifact_weight=halo_artifact_weight,
            texture_blur_kernel=texture_blur_kernel,
            contrast_blur_kernel=contrast_blur_kernel,
            thermal_margin=thermal_margin,
            thermal_temperature=thermal_temperature,
            gradient_overshoot_margin=gradient_overshoot_margin,
        )
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
        fusion_terms = self.fusion_loss(outputs["I_fused"], ir, vis, targets, sam_mask)
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
