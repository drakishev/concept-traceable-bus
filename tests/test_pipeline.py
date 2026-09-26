"""Basic pipeline tests for breast ultrasound VLM.

Tests data loaders, preprocessing, and unified dataset building
without requiring GPU or downloaded data.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal fake dataset directory with synthetic images."""
    # BUSI: benign / malignant / normal subdirs
    for label in ("benign", "malignant", "normal"):
        label_dir = tmp_path / "busi" / label
        label_dir.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 30, i * 30, i * 30))
            img.save(label_dir / f"image_{i:03d}.jpg")

    # BUS-CoT: real format — BUSCoT/BUS-Lesion/trainval/ + BUSCoT/DatasetFiles/
    bus_cot_dir = tmp_path / "bus_cot"
    img_dir = bus_cot_dir / "BUSCoT" / "BUS-Lesion" / "trainval"
    json_dir = bus_cot_dir / "BUSCoT" / "DatasetFiles"
    img_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    annotations = []
    for i in range(4):
        img = Image.new("RGB", (256, 256), color=(100 + i * 10, 100, 100))
        img.save(img_dir / f"{i:06d}@0.png")
        label = 0 if i % 2 == 0 else 1
        annotations.append(
            {
                "query": "Is the lesion benign or malignant?1=Malignant, 0=Benign<image>",
                "images": [f"BUS-Lesion/trainval/{i:06d}@0.png"],
                "response": (
                    f"<reasoning>BIRADS 3. Features consistent with "
                    f"{'benign' if label == 0 else 'malignant'} lesion.</reasoning> "
                    f"<answer> {label} </answer>"
                ),
            }
        )
    with open(json_dir / "vlm_reasoning_trainval.json", "w") as f:
        json.dump(annotations, f)

    return tmp_path


# ─── BUSI loader tests ────────────────────────────────────────────────────────


def test_busi_loader(tmp_dataset_dir: Path) -> None:
    from src.data.datasets.busi import load

    records = load(tmp_dataset_dir / "busi")
    assert len(records) == 9  # 3 labels x 3 images
    for r in records:
        assert "id" in r
        assert "image_path" in r
        assert "conversations" in r
        assert len(r["conversations"]) == 2
        assert r["conversations"][0]["role"] == "user"
        assert r["conversations"][1]["role"] == "assistant"
        assert r["metadata"]["birads"] is not None
        assert Path(r["image_path"]).exists()


def test_busi_loader_missing_dir(tmp_path: Path) -> None:
    from src.data.datasets.busi import load

    records = load(tmp_path / "nonexistent")
    assert records == []


# ─── BUS-CoT loader tests ─────────────────────────────────────────────────────


def test_bus_cot_loader(tmp_dataset_dir: Path) -> None:
    from src.data.datasets.bus_cot import load

    records = load(tmp_dataset_dir / "bus_cot", variant="reasoning", splits=["trainval"])
    assert len(records) == 4
    for r in records:
        assert r["source"] == "bus_cot"
        assert r["id"].startswith("bus_cot_reasoning_trainval_")
        assert "<image>" in r["conversations"][0]["content"]
        assert len(r["conversations"][1]["content"]) > 0
        assert Path(r["image_path"]).exists()


def test_bus_cot_loader_missing_dir(tmp_path: Path) -> None:
    from src.data.datasets.bus_cot import load

    records = load(tmp_path / "nonexistent")
    assert records == []


# ─── Unified dataset tests ────────────────────────────────────────────────────


def test_unified_build(tmp_dataset_dir: Path) -> None:
    from src.data.unified import build

    with tempfile.TemporaryDirectory() as out_dir:
        counts = build(
            raw_root=tmp_dataset_dir,
            out_dir=out_dir,
            sources=["bus_cot", "busi"],
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
        )

        total = counts["train"] + counts["val"] + counts["test"]
        assert total == 13  # 4 bus_cot + 9 busi

        for split in ("train", "val", "test"):
            jsonl_path = Path(out_dir) / f"{split}.jsonl"
            assert jsonl_path.exists(), f"{split}.jsonl not found"
            with open(jsonl_path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) == counts[split]
            for record in lines:
                assert "id" in record
                assert "image_path" in record
                assert "conversations" in record
                assert record["metadata"]["split"] == split


def test_unified_build_no_sources(tmp_path: Path) -> None:
    from src.data.unified import build

    with tempfile.TemporaryDirectory() as out_dir:
        counts = build(raw_root=tmp_path, out_dir=out_dir)
        assert counts == {"train": 0, "val": 0, "test": 0}


# ─── Config tests ─────────────────────────────────────────────────────────────


def test_configs_loadable() -> None:
    from omegaconf import OmegaConf

    config_files = list(Path("configs").rglob("*.yaml"))
    assert len(config_files) > 0, "No YAML configs found"
    for cfg_path in config_files:
        cfg = OmegaConf.load(cfg_path)
        assert cfg is not None, f"Failed to load {cfg_path}"
