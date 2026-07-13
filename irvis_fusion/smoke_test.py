from __future__ import annotations

import argparse

import torch

from .models import IRVISFusionDetectionNet
from .utils.losses import JointFusionDetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick forward/loss smoke test.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=127)
    parser.add_argument("--width", type=int, default=191)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    return parser.parse_args()


def make_targets(batch_size: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
    targets = []
    base_boxes = torch.tensor(
        [[0.30, 0.45, 0.12, 0.18], [0.70, 0.55, 0.10, 0.14]],
        dtype=torch.float32,
        device=device,
    )
    base_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    for _ in range(batch_size):
        targets.append(
            {"boxes": base_boxes.clone(), "labels": base_labels.clone(), "box_format": "cxcywh"}
        )
    return targets


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(7)
    ir = torch.rand(args.batch_size, 1, args.height, args.width, device=device)
    vis = torch.rand(args.batch_size, 1, args.height, args.width, device=device)
    sam = (torch.rand(args.batch_size, 1, args.height, args.width, device=device) > 0.55).float()
    targets = make_targets(args.batch_size, device)

    model = IRVISFusionDetectionNet(
        num_classes=args.num_classes,
        use_sam=not args.no_sam,
        use_feedback=not args.no_feedback,
    ).to(device)
    criterion = JointFusionDetectionLoss(
        num_classes=args.num_classes,
        use_feedback=not args.no_feedback,
    )
    outputs = model(ir, vis, sam_mask=sam, targets=targets)
    losses = criterion(outputs, ir, vis, sam, targets, use_feedback=not args.no_feedback)
    losses["loss"].backward()

    print("smoke_test=ok")
    print(f"I_fused={tuple(outputs['I_fused'].shape)}")
    print(
        "feature_shapes={}".format(
            [tuple(feature.shape) for feature in outputs["fused_features"]]
        )
    )
    print(f"pred_boxes={tuple(outputs['detections']['decoded']['boxes'].shape)}")
    print(f"loss={float(losses['loss'].item()):.4f}")
    print(f"lambda_det={float(losses['lambda_det'].item()):.4f}")
    print(f"detection_recall={float(losses['detection_recall'].item()):.4f}")
    print(f"detection_confidence={float(losses['detection_confidence'].item()):.4f}")
    print(f"forward_logs={outputs['forward_logs']}")


if __name__ == "__main__":
    main()
