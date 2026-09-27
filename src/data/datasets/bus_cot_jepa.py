"""BUS-CoT dataset for VL-JEPA training.

Reads the unified JSONL produced by scripts/preprocess/build_unified.py
and returns (image_tensor, report_text) pairs for contrastive alignment.

Supports two preprocessing modes:
    - "basic": resize → ToTensor → ImageNet normalize
    - "ultrasound": CLAHE contrast enhancement + edge cropping + (train only)
                    horizontal flip / small rotation / brightness-contrast jitter

There is no default mode: a model must be evaluated with the preprocessing it was
trained with, and a silent default once evaluated every batch-7 run on resized
images although training used CLAHE + border crop. `preprocess_mode_of` reads the
mode from a training config, for training and evaluation alike.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Image preprocessing utilities ──────────────────────────────────────────

def _apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0,
                 tile_grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel.

    Standard preprocessing for ultrasound — boosts low-contrast tissue boundaries
    without amplifying global noise.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l_chan)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _crop_dark_borders(img_rgb: np.ndarray, threshold: int = 8) -> np.ndarray:
    """Trim solid black/dark borders surrounding the ultrasound frame.

    Many scans have black borders from the scanner output. This removes them
    so the actual tissue takes up more of the image area.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img_rgb
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_rgb[rmin:rmax + 1, cmin:cmax + 1]
    # only keep crop if it removed ≥10% — avoid aggressive crops on already-clean images
    if (cropped.size > 0
            and cropped.shape[0] * cropped.shape[1]
            < img_rgb.shape[0] * img_rgb.shape[1] * 0.9):
        return cropped
    return img_rgb


def _ultrasound_preprocess(pil_img: Image.Image, apply_clahe: bool = True,
                           crop_borders: bool = True) -> Image.Image:
    """CLAHE + dark-border crop, returns PIL for chaining with torchvision."""
    arr = np.array(pil_img.convert("RGB"))
    if crop_borders:
        arr = _crop_dark_borders(arr)
    if apply_clahe:
        arr = _apply_clahe(arr)
    return Image.fromarray(arr)


# ── Transform builders ────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _basic_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _ultrasound_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Ultrasound-aware preprocessing.

    Train mode adds: horizontal flip, ±10° rotation, brightness/contrast jitter.
    Ultrasound is laterally symmetric so horizontal flips are safe;
    vertical flips are NOT safe (probe orientation matters).
    """
    base = [
        transforms.Lambda(_ultrasound_preprocess),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(base)


def preprocess_mode_of(config) -> str:
    """The preprocessing a training config selects (`data.preprocess_mode`, required).
    `config` is a loaded config or the path of a training YAML."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config) if isinstance(config, (str, Path)) else config
    mode = cfg.data.get("preprocess_mode")
    if mode is None:
        raise ValueError("data.preprocess_mode is not set in the training config")
    return mode


# ── Dataset ───────────────────────────────────────────────────────────────

class BUSCoTJEPADataset(Dataset):
    """Breast Ultrasound Chain-of-Thought dataset for VL-JEPA.

    Each item:
        image  : [3, image_size, image_size] float tensor (ImageNet-normalized)
        report_text : str — the assistant's chain-of-thought reasoning response
        id     : str — sample identifier

    Args:
        jsonl_path: path to the unified JSONL file
        root_dir: path prefix for image_path resolution
        image_size: target square image size
        transform: explicit transform pipeline (instead of preprocess_mode)
        preprocess_mode: "basic" or "ultrasound"; required unless transform is given
        train: when True with preprocess_mode="ultrasound", enables augmentations
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        root_dir: str | Path = ".",
        image_size: int = 224,
        transform: transforms.Compose | None = None,
        preprocess_mode: str | None = None,
        train: bool = False,
    ):
        self.root_dir = Path(root_dir)

        if transform is not None:
            self.transform = transform
        elif preprocess_mode is None:
            raise ValueError("preprocess_mode is required: use the training config's "
                             "(preprocess_mode_of(config))")
        elif preprocess_mode == "ultrasound":
            self.transform = _ultrasound_transform(image_size, train=train)
        elif preprocess_mode == "basic":
            self.transform = _basic_transform(image_size)
        else:
            raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}")

        self.records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        img_path = self.root_dir / rec["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        convs = rec["conversations"]
        report_text = convs[1]["content"] if len(convs) >= 2 else ""

        # extract clinical labels for multi-task auxiliary heads. image_path lets
        # the encoder read BUS-CoT's structured descriptor fields, which cover
        # every paraphrase style (the report text only covers one).
        from src.data.slot_labels import encode_labels
        labels = encode_labels(report_text, rec.get("metadata", {}),
                               image_path=rec.get("image_path"))

        return {
            "image": image,
            "report_text": report_text,
            "id": rec["id"],
            "labels": labels,
        }
