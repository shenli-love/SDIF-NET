"""V2 闭环检测引导融合网络烟雾测试。"""
import sys
sys.path.insert(0, ".")

import torch
from irvis_fusion.models import IRVISFusionDetectionNetV2
from irvis_fusion.utils.losses import JointLossV2

def main():
    print("=" * 60)
    print("SDIF-Net V2 Smoke Test")
    print("=" * 60)

    device = torch.device("cpu")
    model = IRVISFusionDetectionNetV2(
        num_classes=6,
        resnet_base_channels=32,  # 小一点加快测试
        fpn_channels=64,
        use_feedback_loop=True,
    ).to(device)

    criterion = JointLossV2(
        num_classes=6,
        fusion_weight=1.0,
        detection_weight=1.0,
        saliency_weight=0.5,
        modal_weight=0.3,
    )

    # 模拟输入
    B, H, W = 2, 128, 192
    ir = torch.rand(B, 1, H, W, device=device)
    vis = torch.rand(B, 1, H, W, device=device)
    targets = [
        {
            "boxes": torch.tensor([[0.2, 0.3, 0.1, 0.15], [0.6, 0.5, 0.08, 0.12]]),
            "labels": torch.tensor([0, 1]),
            "box_format": "cxcywh",
        },
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "labels": torch.tensor([2]),
            "box_format": "cxcywh",
        },
    ]

    # 训练模式前向传播 (启用反馈回路)
    model.train()
    print("\n[Train Mode - Feedback Loop Enabled]")
    outputs = model(ir, vis, targets=targets, return_logs=True)
    print(f"  I_fused shape: {outputs['I_fused'].shape}")
    print(f"  Feature shapes: {[f.shape for f in outputs['fused_features']]}")
    print(f"  Saliency maps: {[s.shape for s in outputs['saliency_maps']]}")
    print(f"  Modal weights: {[m.shape for m in outputs['modal_weights']]}")
    print(f"  Forward logs: {outputs['forward_logs']}")

    # 计算损失
    losses = criterion(
        outputs, ir, vis, targets,
        ir_features=outputs["ir_features"],
        vis_features=outputs["vis_features"],
    )
    print(f"\n  Loss breakdown:")
    print(f"    total:           {losses['loss'].item():.4f}")
    print(f"    fusion_loss:     {losses['fusion_loss'].item():.4f}")
    print(f"    detection_loss:  {losses['detection_loss'].item():.4f}")
    print(f"    saliency_loss:   {losses['saliency_loss'].item():.4f}")
    print(f"    modal_weight_loss: {losses['modal_weight_loss'].item():.4f}")

    # 反向传播测试
    losses["loss"].backward()
    print("\n  Backward pass: OK")

    # 推理模式前向传播 (禁用反馈回路)
    model.eval()
    print("\n[Eval Mode - Feedback Loop Disabled]")
    with torch.no_grad():
        outputs_eval = model(ir, vis, return_logs=True)
    print(f"  I_fused shape: {outputs_eval['I_fused'].shape}")
    print(f"  Forward logs: {outputs_eval['forward_logs']}")

    print("\n" + "=" * 60)
    print("V2 Smoke Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
