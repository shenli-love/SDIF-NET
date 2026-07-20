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
        description="Train cross-modal QKV detection-feedback IR/VIS fusion."
    )
    parser.add_argument("--data-root", default="datasets/M3FD_Detection")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-size", nargs=2, type=int, default=[768, 1024])
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resnet-base-channels", type=int, default=64)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--anchor-sizes", nargs=4, type=float, default=[8.0, 16.0, 32.0, 64.0])
    parser.add_argument("--modal-specific-weight", type=float, default=0.5)
    parser.add_argument("--small-object-boost", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--detection-weight", type=float, default=1.0)
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs/irvis_sdif_feedback")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", default=None)
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


def load_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[int, int]:
    if checkpoint_path is None:
        return 1, 0
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"resume_missing_keys={len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"resume_unexpected_keys={len(incompatible.unexpected_keys)}")
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            print(f"resume_optimizer_skipped={exc}")
    start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
    global_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0
    print(f"resumed={checkpoint_path} start_epoch={start_epoch} global_step={global_step}")
    return start_epoch, global_step


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
        resnet_base_channels=args.resnet_base_channels,
        fpn_channels=args.fpn_channels,
        anchor_sizes=tuple(args.anchor_sizes),
        use_feedback=not args.no_feedback,
    ).to(device)
    criterion = JointFusionDetectionLoss(
        num_classes=args.num_classes,
        fusion_weight=args.fusion_weight,
        detection_weight=args.detection_weight,
        modal_specific_weight=args.modal_specific_weight,
        small_object_boost=args.small_object_boost,
        use_feedback=not args.no_feedback,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch, global_step = load_training_checkpoint(
        model,
        optimizer,
        args.resume,
        device,
    )
    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        completed_steps = 0
        for step, batch in enumerate(loader, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            targets = move_targets(batch["targets"], device)

            outputs = model(
                ir,
                vis,
                targets=targets,
                return_logs=True,
            )
            losses = criterion(
                outputs,
                ir,
                vis,
                targets,
                use_feedback=not args.no_feedback,
            )
            loss = losses["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            global_step += 1
            completed_steps = step
            running_loss += float(loss.item())
            if step == 1 or step % 20 == 0:
                print(
                    "epoch=[{}/{}] step=[{}/{}] loss={:.4f} fusion={:.4f} "
                    "det={:.4f} intensity={:.4f} grad={:.4f} ssim={:.4f} "
                    "object={:.4f} "
                    "lambda_det={:.3f} recall={:.3f} conf={:.3f}".format(
                        epoch,
                        args.epochs,
                        step,
                        steps_per_epoch,
                        float(losses["loss"].item()),
                        float(losses["fusion_loss"].item()),
                        float(losses["detection_loss"].item()),
                        float(losses["intensity_loss"].item()),
                        float(losses["gradient_loss"].item()),
                        float(losses["ssim_loss"].item()),
                        float(losses["object_loss"].item()),
                        float(losses["lambda_det"].item()),
                        float(losses["detection_recall"].item()),
                        float(losses["detection_confidence"].item()),
                    )
                )

        mean_loss = running_loss / max(completed_steps, 1)
        print(
            f"epoch=[{epoch}/{args.epochs}] mean_loss={mean_loss:.4f} "
            f"steps=[{completed_steps}/{steps_per_epoch}]"
        )
        
        ckpt_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        
        latest_path = output_dir / "latest.pt"
        torch.save(ckpt_data, latest_path)
        
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = output_dir / "best.pt"
            torch.save(ckpt_data, best_path)
            print(f"saved_best={best_path} loss={best_loss:.4f}")
        
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt_data, ckpt_path)
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
