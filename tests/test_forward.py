from __future__ import annotations

import torch

from irvis_fusion.models import IRVISFusionDetectionNet
from irvis_fusion.utils.losses import JointFusionDetectionLoss


def test_forward_backward_dynamic_shape() -> None:
    batch_size = 1
    height, width = 31, 47
    ir = torch.rand(batch_size, 1, height, width)
    vis = torch.rand(batch_size, 1, height, width)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "box_format": "cxcywh",
        }
        for _ in range(batch_size)
    ]
    model = IRVISFusionDetectionNet(
        num_classes=6,
        resnet_base_channels=8,
        fpn_channels=32,
    )
    criterion = JointFusionDetectionLoss(num_classes=6)
    outputs = model(ir, vis, targets=targets)
    losses = criterion(outputs, ir, vis, targets)
    losses["loss"].backward()
    assert outputs["I_fused"].shape == ir.shape
    assert len(outputs["fused_features"]) == 4
    assert outputs["detections"]["decoded"]["boxes"].shape[0] == batch_size
