from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import M3FDDataset, detection_collate
from .models import IRVISFusionDetectionNet
from .utils.losses import JointFusionDetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAM-guided detection-feedback IR/VIS fusion."
    )
    parser.add_argument("--data-root", default="datasets/M3FD_Detection")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-size", nargs=2, type=int, default=[256, 320])
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--detector-backend", default="yolo_like", choices=["yolo_like", "ultralytics"])
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--detection-weight", type=float, default=1.0)
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs/irvis_sdif_feedback")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def move_targets(
    targets: list[dict[str, torch.Tensor | str]],
    device: torch.device,
) -> list[dict[str, torch.Tensor | str]]:
    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return moved


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(args.image_size)
    dataset = M3FDDataset(args.data_root, split=args.split, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )

    model = IRVISFusionDetectionNet(
        num_classes=args.num_classes,
        use_sam=not args.no_sam,
        use_feedback=not args.no_feedback,
        detector_backend=args.detector_backend,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
    ).to(device)
    criterion = JointFusionDetectionLoss(
        num_classes=args.num_classes,
        fusion_weight=args.fusion_weight,
        detection_weight=args.detection_weight,
        use_feedback=not args.no_feedback,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            sam = batch["sam_mask"].to(device, non_blocking=True)
            targets = move_targets(batch["targets"], device)

            outputs = model(
                ir,
                vis,
                sam_mask=sam,
                targets=targets,
                return_logs=True,
            )
            losses = criterion(
                outputs,
                ir,
                vis,
                sam,
                targets,
                use_feedback=not args.no_feedback,
            )
            loss = losses["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            global_step += 1
            running_loss += float(loss.item())
            if step == 1 or step % 20 == 0:
                print(
                    "epoch={:03d} step={:05d} loss={:.4f} fusion={:.4f} "
                    "det={:.4f} lambda_det={:.3f} recall={:.3f} conf={:.3f}".format(
                        epoch,
                        step,
                        float(losses["loss"].item()),
                        float(losses["fusion_loss"].item()),
                        float(losses["detection_loss"].item()),
                        float(losses["lambda_det"].item()),
                        float(losses["detection_recall"].item()),
                        float(losses["detection_confidence"].item()),
                    )
                )

        mean_loss = running_loss / max(len(loader), 1)
        print(f"epoch={epoch:03d} mean_loss={mean_loss:.4f}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
