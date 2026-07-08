from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .data import M3FDDataset, detection_collate
from .models import IRVISFusionDetectionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IR/VIS fusion inference and save fused images/detections."
    )
    parser.add_argument("--data-root", default="datasets/M3FD_Detection")
    parser.add_argument("--split", default="val")
    parser.add_argument("--ir-image", default=None)
    parser.add_argument("--vis-image", default=None)
    parser.add_argument("--sam-mask", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="runs/infer")
    parser.add_argument("--output-image", default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=[768, 1024])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--detector-backend", default="yolo_like", choices=["yolo_like", "ultralytics"])
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=1024)
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--yolo-max-det", type=int, default=500)
    parser.add_argument("--save-conf", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | None, device: torch.device) -> None:
    if checkpoint_path is None:
        print("checkpoint=None; running with current model weights.")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = model.load_state_dict(state, strict=False)
    print(f"loaded_checkpoint={checkpoint_path}")
    if incompatible.missing_keys:
        print(f"missing_keys={len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"unexpected_keys={len(incompatible.unexpected_keys)}")


def tensor_to_gray_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().clamp(0.0, 1.0).cpu()
    if image.dim() == 3:
        image = image.squeeze(0)
    array = (image.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="L")


def load_gray_tensor(path: str | Path, image_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("L")
    height, width = image_size
    if image.size != (width, height):
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((width, height), resample)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def draw_detections(
    fused: Image.Image,
    result: dict[str, torch.Tensor],
    conf_threshold: float,
) -> Image.Image:
    canvas = fused.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result["labels"].detach().cpu()
    for box, score, label in zip(boxes, scores, labels):
        score_value = float(score.item())
        if score_value < conf_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        xyxy = [
            max(0, min(width - 1, int(round(x1 * width)))),
            max(0, min(height - 1, int(round(y1 * height)))),
            max(0, min(width - 1, int(round(x2 * width)))),
            max(0, min(height - 1, int(round(y2 * height)))),
        ]
        draw.rectangle(xyxy, outline=(255, 64, 64), width=2)
        draw.text((xyxy[0], max(0, xyxy[1] - 12)), f"{int(label)} {score_value:.2f}", fill=(255, 64, 64))
    return canvas


def save_detection_txt(path: Path, result: dict[str, torch.Tensor], conf_threshold: float) -> None:
    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result["labels"].detach().cpu()
    lines = []
    for box, score, label in zip(boxes, scores, labels):
        score_value = float(score.item())
        if score_value < conf_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        lines.append(
            f"{int(label)} {score_value:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@torch.no_grad()
def infer_single_image_pair(
    args: argparse.Namespace,
    model: IRVISFusionDetectionNet,
    device: torch.device,
) -> None:
    if args.ir_image is None or args.vis_image is None:
        raise ValueError("--ir-image and --vis-image must be provided together.")

    image_size = tuple(args.image_size)
    ir = load_gray_tensor(args.ir_image, image_size).to(device)
    vis = load_gray_tensor(args.vis_image, image_size).to(device)
    sam = None
    if args.sam_mask is not None:
        sam = load_gray_tensor(args.sam_mask, image_size).to(device)

    outputs = model(ir, vis, sam_mask=sam, return_logs=False)
    fused_image = tensor_to_gray_image(outputs["I_fused"][0])
    output_image = (
        Path(args.output_image)
        if args.output_image is not None
        else Path(args.output_dir) / "single_fused.png"
    )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    fused_image.save(output_image)
    print(f"saved={output_image}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = IRVISFusionDetectionNet(
        num_classes=args.num_classes,
        use_sam=not args.no_sam,
        detector_backend=args.detector_backend,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if args.ir_image is not None or args.vis_image is not None:
        infer_single_image_pair(args, model, device)
        return

    output_dir = Path(args.output_dir)
    fused_dir = output_dir / "fused"
    det_dir = output_dir / "detections"
    vis_dir = output_dir / "visualizations"
    fused_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = M3FDDataset(
        args.data_root,
        split=args.split,
        image_size=tuple(args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )

    saved = 0
    for batch in loader:
        ir = batch["ir"].to(device, non_blocking=True)
        vis = batch["vis"].to(device, non_blocking=True)
        sam = batch["sam_mask"].to(device, non_blocking=True)
        outputs = model(ir, vis, sam_mask=sam, return_logs=False)
        fused = outputs["I_fused"]
        decoded = outputs["detections"]["decoded"]
        results = model.detector.postprocess(decoded, conf_threshold=args.save_conf)

        for batch_idx, result in enumerate(results):
            target = batch["targets"][batch_idx]
            sample_id = str(target.get("image_id", f"{saved:06d}"))
            fused_image = tensor_to_gray_image(fused[batch_idx])
            fused_image.save(fused_dir / f"{sample_id}.png")
            save_detection_txt(det_dir / f"{sample_id}.txt", result, args.save_conf)
            draw_detections(fused_image, result, args.save_conf).save(vis_dir / f"{sample_id}.png")
            saved += 1
            if args.max_samples > 0 and saved >= args.max_samples:
                print(f"saved={saved} output_dir={output_dir}")
                return

    print(f"saved={saved} output_dir={output_dir}")


if __name__ == "__main__":
    main()
