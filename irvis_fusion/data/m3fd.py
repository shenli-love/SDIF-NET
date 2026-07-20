from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _load_grayscale(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _resize_image(image: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None:
        return image
    if image.shape[-2:] == size:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _read_yolo_label(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = []
    labels = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                labels.append(int(float(cls)))
                boxes.append([float(cx), float(cy), float(w), float(h)])
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0,), dtype=torch.long
        )
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class M3FDDataset(Dataset):
    """M3FD IR/VIS/YOLO dataset reader."""

    def __init__(
        self,
        root: str | Path = "datasets/M3FD_Detection",
        split: str = "train",
        image_size: tuple[int, int] | None = (768, 1024),
        ir_dir: str = "ir",
        vis_dir: str = "vi",
        labels_dir: str = "labels",
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        meta_path = self.root / "meta" / f"{split}.txt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing split file: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            self.ids = [line.strip() for line in handle if line.strip()]
        self.ir_root = self.root / ir_dir
        self.vis_root = self.root / vis_dir
        self.labels_root = self.root / labels_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        ir = _resize_image(_load_grayscale(self.ir_root / f"{sample_id}.png"), self.image_size)
        vis = _resize_image(
            _load_grayscale(self.vis_root / f"{sample_id}.png"), self.image_size
        )
        boxes, labels = _read_yolo_label(self.labels_root / f"{sample_id}.txt")
        target = {
            "boxes": boxes,
            "labels": labels,
            "box_format": "cxcywh",
            "image_id": sample_id,
        }
        return {"ir": ir, "vis": vis, "target": target}


def detection_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "ir": torch.stack([item["ir"] for item in batch], dim=0),
        "vis": torch.stack([item["vis"] for item in batch], dim=0),
        "targets": [item["target"] for item in batch],
    }
