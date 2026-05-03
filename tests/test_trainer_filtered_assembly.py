"""
tests/test_trainer_filtered_assembly.py
========================================
Trainer integration test for filtered-COCO datasets.

This test verifies that YOLOTrainer._prepare_dataset() correctly handles:
  - 5-class filtered dataset (not 80-class COCO default)
  - From-scratch model (yolov8s.yaml, pretrained=False)
  - Filtered class_names propagated into dataset.yaml
  - Dense class IDs (0..4) survive end-to-end through label copying
  - inner_val partition assembled separately, never receiving pseudo-labels

This does NOT run model.train() — purely dataset assembly.
That keeps the test fast (~1 second) and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # type: ignore

from autolabel.training.trainer import TrainingConfig, YOLOTrainer
from autolabel.bank.schemas import PseudoLabel


VEHICLE_CLASS_NAMES = ["bicycle", "car", "motorcycle", "bus", "truck"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_min_jpg(path: Path) -> None:
    """Write a minimal valid JPEG so cv2.imread can decode dimensions."""
    jpeg = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb0043000806060706050806"
        "0707080909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e272022"
        "2c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b0801000100"
        "0101110000ffc4001f0000010501010101010100000000000000000102030405060708090a0b"
        "ffc4001500010100000000000000000000000000000000ffda0008010100003f00d2cf20ffd9"
    )
    path.write_bytes(jpeg)


def _build_filtered_tree(root: Path,
                          n_train:     int = 8,
                          n_inner_val: int = 2,
                          n_unlabeled: int = 4) -> Tuple[Path, Path, dict]:
    """
    Build a synthetic 'filtered COCO' directory layout:
        root/images/train2017/{NNNN}.jpg
        root/labels/train2017/{NNNN}.txt   (dense class IDs 0..4)

    Returns (image_dir, label_dir, labeled_data_dict) ready to pass to
    YOLOTrainer._prepare_dataset.
    """
    image_dir = root / "images" / "train2017"
    label_dir = root / "labels" / "train2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    total = n_train + n_inner_val + n_unlabeled
    train_fnames     = []
    inner_val_fnames = []
    unlabeled_fnames = []

    # Cycle through dense class IDs 0..4 so every class appears at least once
    for i in range(total):
        image_id = 100 + i
        fname    = f"{image_id:012d}.jpg"
        cid      = i % 5  # cycle 0..4

        # Image
        _write_min_jpg(image_dir / fname)
        # Label (dense class ID, valid YOLO bbox)
        (label_dir / f"{image_id:012d}.txt").write_text(
            f"{cid} 0.5 0.5 0.4 0.4\n"
        )

        if i < n_train:
            train_fnames.append(fname)
        elif i < n_train + n_inner_val:
            inner_val_fnames.append(fname)
        else:
            unlabeled_fnames.append(fname)

    labeled_data = {
        "image_dir":            str(image_dir),
        "label_dir":            str(label_dir),
        "image_list":           train_fnames,
        "inner_val_image_list": inner_val_fnames,
        "unlabeled_image_dir":  str(image_dir),  # same-dir COCO case
        "unlabeled_list":       unlabeled_fnames,
    }
    return image_dir, label_dir, labeled_data


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def filtered_trainer(tmp_path):
    """
    YOLOTrainer configured for the from-scratch filtered-COCO experiment.
    Uses yolov8s.yaml with pretrained=False, nc=5, vehicle class names.
    """
    cfg = TrainingConfig(
        epochs      = 1,
        batch       = 1,
        device      = "cpu",
        workers     = 0,
        pretrained  = False,
        base_model  = "yolov8s.yaml",   # from scratch
        imgsz       = 64,               # tiny — assembly only, no training
        output_dir  = str(tmp_path / "out"),
        num_classes = 5,
        class_names = VEHICLE_CLASS_NAMES,
        patience    = 5,
        optimizer   = "AdamW",
        lr0         = 0.001,
    )
    return YOLOTrainer(cfg)


@pytest.fixture
def filtered_data(tmp_path):
    """A synthetic filtered-COCO labeled_data dict with realistic counts."""
    return _build_filtered_tree(tmp_path / "filtered")[2]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_dataset_yaml_has_nc_5(filtered_trainer, filtered_data, tmp_path):
    """dataset.yaml must reflect 5 classes, not the 80-class default."""
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    ds = yaml.safe_load((round_dir / "dataset.yaml").read_text())
    assert ds["nc"] == 5, f"expected nc=5, got nc={ds['nc']}"


def test_dataset_yaml_has_vehicle_names(filtered_trainer, filtered_data):
    """dataset.yaml `names` must list the 5 vehicle classes in order."""
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    ds = yaml.safe_load((round_dir / "dataset.yaml").read_text())
    assert ds["names"] == VEHICLE_CLASS_NAMES, \
        f"expected {VEHICLE_CLASS_NAMES}, got {ds['names']}"


def test_dataset_yaml_val_routes_to_inner_val(filtered_trainer, filtered_data):
    """`val:` must point at images/val (the inner-val partition), not train."""
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    ds = yaml.safe_load((round_dir / "dataset.yaml").read_text())
    assert ds["val"] == "images/val", \
        f"expected val=images/val (inner-val partition), got {ds['val']}"
    assert ds["train"] == "images/train"


def test_train_partition_assembled(filtered_trainer, filtered_data):
    """All labeled-train images + labels must land under images/train + labels/train."""
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    img_train   = round_dir / "images" / "train"
    label_train = round_dir / "labels" / "train"

    expected_imgs = sorted(filtered_data["image_list"])
    actual_imgs   = sorted(p.name for p in img_train.iterdir())
    assert actual_imgs == expected_imgs

    expected_labels = sorted(Path(f).stem + ".txt"
                              for f in filtered_data["image_list"])
    actual_labels   = sorted(p.name for p in label_train.iterdir())
    assert actual_labels == expected_labels


def test_inner_val_partition_assembled(filtered_trainer, filtered_data):
    """All inner-val images + labels must land under images/val + labels/val."""
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    img_val   = round_dir / "images" / "val"
    label_val = round_dir / "labels" / "val"

    expected_imgs = sorted(filtered_data["inner_val_image_list"])
    actual_imgs   = sorted(p.name for p in img_val.iterdir())
    assert actual_imgs == expected_imgs

    expected_labels = sorted(Path(f).stem + ".txt"
                              for f in filtered_data["inner_val_image_list"])
    actual_labels   = sorted(p.name for p in label_val.iterdir())
    assert actual_labels == expected_labels


def test_class_ids_dense_in_assembled_labels(filtered_trainer, filtered_data):
    """
    Class IDs in the assembled label files must be dense {0..4}.
    Catches any regression where class IDs got remapped to 80-class
    space or back to sparse COCO IDs during assembly.
    """
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    seen_ids = set()
    for label_file in (round_dir / "labels" / "train").iterdir():
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            seen_ids.add(cid)
    for label_file in (round_dir / "labels" / "val").iterdir():
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            seen_ids.add(cid)

    assert seen_ids.issubset({0, 1, 2, 3, 4}), \
        f"expected class IDs in {{0,1,2,3,4}}, got {sorted(seen_ids)}"
    # All 5 should be present given our synthetic data cycles 0..4
    assert seen_ids == {0, 1, 2, 3, 4}, \
        f"expected all 5 classes present, got {sorted(seen_ids)}"


def test_pseudo_label_appended_to_train_only(filtered_trainer,
                                                filtered_data,
                                                tmp_path):
    """
    Pseudo-labels must land in images/train + labels/train, never in
    images/val or labels/val. Inner val must remain frozen ground truth.
    """
    # Pick the first unlabeled image as the pseudo-label target
    target_fname = filtered_data["unlabeled_list"][0]
    target_image_id = Path(target_fname).stem

    pl = PseudoLabel(
        image_id     = target_image_id,
        box_id       = "test_pl",
        round_id     = 1,
        box          = (10.0, 10.0, 30.0, 30.0),  # absolute pixels
        pred_class   = 2,                          # motorcycle (dense yolo id)
        class_scores = (0.0, 0.0, 1.0, 0.0, 0.0),  # one-hot at idx 2
        confidence   = 0.95,
    )

    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[pl], round_id=1
    )

    # Pseudo image should be in train, NOT val
    img_train_files = {p.name for p in (round_dir / "images" / "train").iterdir()}
    img_val_files   = {p.name for p in (round_dir / "images" / "val").iterdir()}
    assert target_fname in img_train_files
    assert target_fname not in img_val_files


def test_from_scratch_config_does_not_break_assembly(filtered_trainer,
                                                       filtered_data):
    """
    Setting pretrained=False and base_model=yolov8s.yaml must not break
    dataset assembly. (This is a smoke check — the actual training would
    fail without a YOLO install, but assembly should not depend on it.)
    """
    assert filtered_trainer.config.pretrained is False
    assert filtered_trainer.config.base_model == "yolov8s.yaml"
    # Should complete without raising
    round_dir = filtered_trainer._prepare_dataset(
        filtered_data, pseudo_labels=[], round_id=1
    )
    assert (round_dir / "dataset.yaml").exists()