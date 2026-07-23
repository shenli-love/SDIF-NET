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


class MaxRuleFusionLoss(nn.Module):
    """基于最大值规则的融合损失。

    核心思想：融合图像在每个像素/梯度上应至少达到两个源图像的最大值，
    而不是简单复制某一个模态。这确保：
    - IR 的热对比目标被保留
    - VIS 的纹理边缘被保留
    - 网络有动力去"超越"任何单一模态
    """

    def __init__(
        self,
        ssim_window: int = 7,
        intensity_weight: float = 1.0,
        gradient_weight: float = 3.0,
        ssim_weight: float = 1.0,
        max_rule_weight: float = 2.0,
        object_enhance_weight: float = 1.5,
    ) -> None:
        super().__init__()
        self.ssim_window = ssim_window
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.max_rule_weight = max_rule_weight
        self.object_enhance_weight = object_enhance_weight

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        detections: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        object_map = self._build_weight_map(targets, fused.shape, fused.device, fused.dtype)

        # === 1. 强度损失：fused 应接近 max(ir, vis) 而非单一模态 ===
        max_intensity = torch.maximum(ir, vis)
        intensity_loss = charbonnier(fused - max_intensity).mean()

        # === 2. 梯度损失：fused 梯度应 >= max(ir_grad, vis_grad) ===
        fused_grad = gradient_map(fused)
        ir_grad = gradient_map(ir)
        vis_grad = gradient_map(vis)
        max_grad = torch.maximum(ir_grad, vis_grad)

        # 只惩罚 fused_grad < max_grad 的地方 (hinge loss)
        gradient_deficit = F.relu(max_grad - fused_grad)
        gradient_loss = gradient_deficit.mean()

        # 额外：fused 梯度与 max 梯度的 L1
        gradient_l1 = charbonnier(fused_grad - max_grad).mean()

        # === 3. SSIM 损失：与两个源都保持结构相似 ===
        ssim_ir = self._ssim_loss(fused, ir)
        ssim_vis = self._ssim_loss(fused, vis)
        ssim_loss = (ssim_ir + ssim_vis) / 2.0

        # === 4. 目标区域增强：目标区域梯度要更强 ===
        object_gradient_loss = (gradient_deficit * object_map).sum() / object_map.sum().clamp_min(1.0)

        # === 5. 对比度保持：融合图像的标准差不应低于源 ===
        contrast_loss = F.relu(ir.std() - fused.std()) + F.relu(vis.std() - fused.std())

        total = (
            self.intensity_weight * intensity_loss
            + self.gradient_weight * (gradient_loss + 0.5 * gradient_l1)
            + self.ssim_weight * ssim_loss
            + self.max_rule_weight * object_gradient_loss * self.object_enhance_weight
            + 0.5 * contrast_loss
        )

        zero = fused.new_tensor(0.0)
        return {
            "fusion_loss": total,
            "intensity_loss": intensity_loss,
            "gradient_loss": gradient_loss + gradient_l1,
            "ssim_loss": ssim_loss,
            "object_loss": object_gradient_loss,
            "contrast_loss": contrast_loss,
            "nan_guard": zero,
        }

    def _ssim_loss(self, fused: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.ssim_window
        padding = window // 2
        mu_f = F.avg_pool2d(fused, window, 1, padding, count_include_pad=False)
        mu_t = F.avg_pool2d(target, window, 1, padding, count_include_pad=False)
        sig_f = F.avg_pool2d(fused * fused, window, 1, padding, count_include_pad=False) - mu_f.pow(2)
        sig_t = F.avg_pool2d(target * target, window, 1, padding, count_include_pad=False) - mu_t.pow(2)
        sig_cross = F.avg_pool2d(fused * target, window, 1, padding, count_include_pad=False) - mu_f * mu_t
        sig_f = sig_f.clamp_min(0.0)
        sig_t = sig_t.clamp_min(0.0)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_f * mu_t + c1) * (2 * sig_cross + c2)) / (
            (mu_f.pow(2) + mu_t.pow(2) + c1) * (sig_f + sig_t + c2)
        ).clamp_min(1e-6)
        return (1.0 - ssim.clamp(0.0, 1.0)).mean()

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
                left = max(0, min(width - 1, int(x1 * width)))
                top = max(0, min(height - 1, int(y1 * height)))
                right = max(left + 1, min(width, int(x2 * width)))
                bottom = max(top + 1, min(height, int(y2 * height)))
                mask[batch_idx, :, top:bottom, left:right] = 1.0
        return mask


# 保持向后兼容
DetectionAwareFusionLoss = MaxRuleFusionLoss
FusionReconstructionLoss = MaxRuleFusionLoss


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
        gradient_weight: float = 3.0,
        ssim_weight: float = 1.0,
        max_rule_weight: float = 2.0,
        small_object_boost: float = 1.0,
        # 保留这些参数以兼容命令行，但不再使用
        object_weight: float = 1.5,
        background_weight: float = 1.5,
        modal_specific_weight: float = 0.8,
    ) -> None:
        super().__init__()
        self.fusion_weight = fusion_weight
        self.detection_weight = detection_weight
        self.use_feedback = use_feedback
        self.fusion_loss = MaxRuleFusionLoss(
            gradient_weight=gradient_weight,
            ssim_weight=ssim_weight,
            max_rule_weight=max_rule_weight,
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


# =============================================================================
# 新版：显著性监督损失 + 闭环联合损失
# =============================================================================


class SaliencySupervisionLoss(nn.Module):
    """用 GT 框生成高斯显著性图，监督 TaskSaliencyPredictor。

    同时用 GT 框内 IR/VIS 的特征能量比来监督 modal_weight，
    确保模态权重预测器学到“哪个模态在目标区域更有信息量”。
    """

    def __init__(self, sigma_factor: float = 0.25) -> None:
        super().__init__()
        self.sigma_factor = sigma_factor

    def forward(
        self,
        saliency_maps: tuple[torch.Tensor, ...],
        modal_weights: tuple[torch.Tensor, ...],
        ir_features: tuple[torch.Tensor, ...],
        vis_features: tuple[torch.Tensor, ...],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        device = saliency_maps[0].device
        dtype = saliency_maps[0].dtype
        total_sal_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_modal_loss = torch.tensor(0.0, device=device, dtype=dtype)
        num_scales = len(saliency_maps)

        for scale_idx in range(num_scales):
            pred_sal = saliency_maps[scale_idx]
            B, _, H, W = pred_sal.shape

            # 生成 GT 高斯显著性图
            gt_sal = self._build_gaussian_saliency(targets, B, H, W, device, dtype)
            total_sal_loss = total_sal_loss + F.binary_cross_entropy(
                pred_sal.clamp(1e-6, 1 - 1e-6), gt_sal, reduction="mean"
            )

            # 生成模态权重监督
            pred_mw = modal_weights[scale_idx]
            gt_mw = self._build_modal_weight(
                ir_features[scale_idx], vis_features[scale_idx], B, H, W
            )
            total_modal_loss = total_modal_loss + F.l1_loss(pred_mw, gt_mw)

        return {
            "saliency_loss": total_sal_loss / num_scales,
            "modal_weight_loss": total_modal_loss / num_scales,
        }

    def _build_gaussian_saliency(
        self,
        targets: list[dict[str, torch.Tensor]],
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """GT 框中心生成高斯响应图作为显著性监督。"""
        sal = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        for b, target in enumerate(targets):
            if b >= B:
                break
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                continue
            if target.get("box_format", "cxcywh") == "cxcywh":
                boxes = cxcywh_to_xyxy(boxes)
            boxes = boxes.clamp(0, 1)
            for box in boxes:
                cx = ((box[0] + box[2]) / 2 * W).long().clamp(0, W - 1)
                cy = ((box[1] + box[3]) / 2 * H).long().clamp(0, H - 1)
                bw = ((box[2] - box[0]) * W).clamp(min=1)
                bh = ((box[3] - box[1]) * H).clamp(min=1)
                sigma = max(float(bw), float(bh)) * self.sigma_factor
                sigma = max(sigma, 1e-3)
                yy = torch.arange(H, device=device, dtype=dtype)
                xx = torch.arange(W, device=device, dtype=dtype)
                gy = torch.exp(-0.5 * ((yy - cy.float()) / sigma) ** 2)
                gx = torch.exp(-0.5 * ((xx - cx.float()) / sigma) ** 2)
                gaussian = gy.unsqueeze(1) * gx.unsqueeze(0)
                sal[b, 0] = torch.maximum(sal[b, 0], gaussian)
        return sal.clamp(0, 1)

    def _build_modal_weight(
        self,
        ir_feat: torch.Tensor,
        vis_feat: torch.Tensor,
        B: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """基于特征能量比生成模态权重监督。"""
        device = ir_feat.device
        dtype = ir_feat.dtype
        # detach 避免梯度回传到编码器
        ir_energy = ir_feat.detach().abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        vis_energy = vis_feat.detach().abs().mean(dim=1, keepdim=True)
        total_e = ir_energy + vis_energy + 1e-6
        mw = torch.zeros(B, 2, H, W, device=device, dtype=dtype)
        mw[:, 0:1] = ir_energy / total_e
        mw[:, 1:2] = vis_energy / total_e
        return mw


class JointLossV2(nn.Module):
    """闭环检测引导融合网络的联合损失。

    组成:
        L_total = w_fusion * L_fusion
                + w_det * L_detection
                + w_sal * L_saliency
                + w_modal * L_modal_weight

    与旧版 JointFusionDetectionLoss 的区别:
    - 移除了 DetectionFeedbackLossWeight (λ_det 标量机制)
    - 新增 SaliencySupervisionLoss 确保显著性预测器学到检测语义
    - 检测引导已在特征层面实现，不再需要损失层面的间接调节
    """

    def __init__(
        self,
        num_classes: int,
        fusion_weight: float = 1.0,
        detection_weight: float = 1.0,
        saliency_weight: float = 0.5,
        modal_weight: float = 0.3,
        gradient_weight: float = 3.0,
        ssim_weight: float = 1.0,
        max_rule_weight: float = 2.0,
        small_object_boost: float = 1.0,
    ) -> None:
        super().__init__()
        self.fusion_weight = fusion_weight
        self.detection_weight = detection_weight
        self.saliency_weight = saliency_weight
        self.modal_weight_coef = modal_weight

        self.fusion_loss = MaxRuleFusionLoss(
            gradient_weight=gradient_weight,
            ssim_weight=ssim_weight,
            max_rule_weight=max_rule_weight,
        )
        self.detection_loss = YOLOLikeDetectionLoss(
            num_classes=num_classes,
            small_object_boost=small_object_boost,
        )
        self.saliency_loss = SaliencySupervisionLoss()

    def forward(
        self,
        outputs: dict[str, object],
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        ir_features: tuple[torch.Tensor, ...] | None = None,
        vis_features: tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        detections = outputs["detections"]

        # 融合重建损失
        fusion_terms = self.fusion_loss(
            outputs["I_fused"], ir, vis, targets, detections=detections
        )

        # 检测损失
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

        # 显著性监督损失
        if ir_features is not None and vis_features is not None:
            saliency_terms = self.saliency_loss(
                outputs["saliency_maps"],
                outputs["modal_weights"],
                ir_features,
                vis_features,
                targets,
            )
        else:
            zero = outputs["I_fused"].new_tensor(0.0)
            saliency_terms = {"saliency_loss": zero, "modal_weight_loss": zero}

        total = (
            self.fusion_weight * fusion_terms["fusion_loss"]
            + self.detection_weight * detection_terms["detection_loss"]
            + self.saliency_weight * saliency_terms["saliency_loss"]
            + self.modal_weight_coef * saliency_terms["modal_weight_loss"]
        )

        return {
            "loss": total,
            **fusion_terms,
            **detection_terms,
            **saliency_terms,
        }
