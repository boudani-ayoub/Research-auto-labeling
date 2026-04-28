"""
tests/test_trainer_assembly.py
================================
Phase 6B tests for training/trainer.py — dataset assembly only.
No real YOLO training. No GPU. No COCO.

Tests cover:
  - dataset.yaml written with correct fields
  - images/train and labels/train directories created
  - labeled images and labels copied
  - pseudo-labels from UNLABELED images copied and written (the real case)
  - pseudo-labels from labeled images appended to existing label files
  - YOLO format normalisation correctness
  - degenerate box skipped
  - round directory naming convention
  - unresolvable pseudo-label image raises FileNotFoundError (not silent skip)
  - filename collision between labeled and unlabeled images handled
  - string-path labeled_data input resolves correctly
  - all coordinates normalised to [0,1]
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import PseudoLabel
from autolabel.training.trainer import (
    TrainingConfig, YOLOTrainer, assemble_dataset_only,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_scores(pred_class: int, nc: int = 80) -> tuple:
    s = [0.0] * nc; s[pred_class] = 1.0
    return tuple(s)


def make_pl(image_id, box_id=None, round_id=1,
            box=(100.0, 100.0, 300.0, 300.0),
            pred_class=0, confidence=0.8) -> PseudoLabel:
    if box_id is None:
        box_id = f"{image_id}_r{round_id}_0000"
    return PseudoLabel(
        image_id=image_id, box_id=box_id, round_id=round_id,
        box=box, pred_class=pred_class,
        class_scores=make_scores(pred_class), confidence=confidence,
    )


def make_image(path: Path, h: int = 480, w: int = 640) -> None:
    """Write a minimal JPEG to disk."""
    import cv2
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def make_labeled_dataset(tmp_dir: Path,
                          n_images:       int = 3,
                          unlabeled_count: int = 2) -> dict:
    """
    Create a labeled dataset and an unlabeled image directory.
    Returns a labeled_data dict with unlabeled_image_dir set.
    """
    img_dir   = tmp_dir / "labeled" / "images" / "train"
    label_dir = tmp_dir / "labeled" / "labels" / "train"
    unlab_dir = tmp_dir / "unlabeled"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    unlab_dir.mkdir(parents=True)

    image_list = []
    for i in range(n_images):
        fname = f"labeled_{i:04d}.jpg"
        make_image(img_dir / fname)
        with open(label_dir / f"labeled_{i:04d}.txt", "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")
        image_list.append(fname)

    for i in range(unlabeled_count):
        make_image(unlab_dir / f"unlabeled_{i:04d}.jpg")

    return {
        "image_dir":           str(img_dir),
        "label_dir":           str(label_dir),
        "image_list":          image_list,
        "unlabeled_image_dir": str(unlab_dir),
    }


def default_config(output_dir: str) -> TrainingConfig:
    return TrainingConfig(
        epochs=1, batch=1, device="cpu", workers=0,
        pretrained=True, base_model="yolov8m.pt",
        imgsz=640, output_dir=output_dir, num_classes=80,
    )


# ── YAML tests ────────────────────────────────────────────────────────────────

def test_yaml_written():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp)
        cfg     = default_config(str(tmp / "out"))
        rd = assemble_dataset_only(labeled, [], 0, cfg)
        assert (rd / "dataset.yaml").exists()
    print("PASS  test_yaml_written")


def test_yaml_fields():
    import yaml as yaml_mod
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp)
        cfg     = default_config(str(tmp / "out"))
        rd = assemble_dataset_only(labeled, [], 0, cfg)
        data = yaml_mod.safe_load((rd / "dataset.yaml").read_text())
        assert "path" in data and "train" in data
        assert data["nc"] == 80
        assert len(data["names"]) == 80
    print("PASS  test_yaml_fields")


# ── Directory structure ───────────────────────────────────────────────────────

def test_dirs_created():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp)
        cfg     = default_config(str(tmp / "out"))
        rd = assemble_dataset_only(labeled, [], 0, cfg)
        assert (rd / "images" / "train").is_dir()
        assert (rd / "labels" / "train").is_dir()
    print("PASS  test_dirs_created")


# ── Labeled data copy ─────────────────────────────────────────────────────────

def test_labeled_images_and_labels_copied():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=3)
        cfg     = default_config(str(tmp / "out"))
        rd = assemble_dataset_only(labeled, [], 0, cfg)
        for fname in labeled["image_list"]:
            assert (rd / "images" / "train" / fname).exists()
            assert (rd / "labels" / "train" / (Path(fname).stem + ".txt")).exists()
    print("PASS  test_labeled_images_and_labels_copied")


# ── Pseudo-label from unlabeled image (the real case) ────────────────────────

def test_pseudo_label_from_unlabeled_image_copied():
    """
    Core case: pseudo-label belongs to an unlabeled image.
    Image must be copied from unlabeled_image_dir and label file written.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=2, unlabeled_count=2)
        cfg     = default_config(str(tmp / "out"))

        # Pseudo-label on unlabeled image (not in labeled image_list)
        pl = make_pl("unlabeled_0000", box=(50.0, 50.0, 200.0, 200.0), pred_class=7)
        rd = assemble_dataset_only(labeled, [pl], 1, cfg)

        # Image must be copied
        img_dst = rd / "images" / "train" / "unlabeled_0000.jpg"
        assert img_dst.exists(), "Unlabeled image not copied to dataset"

        # Label file must be written
        label_dst = rd / "labels" / "train" / "unlabeled_0000.txt"
        assert label_dst.exists(), "Label file not written for unlabeled image"
        lines = [l for l in label_dst.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("7 ")
    print("PASS  test_pseudo_label_from_unlabeled_image_copied")


def test_pseudo_label_appended_to_labeled_image():
    """Pseudo-label on a labeled image appends to existing label file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=1)
        cfg     = default_config(str(tmp / "out"))

        fname    = labeled["image_list"][0]
        image_id = Path(fname).stem
        pl = make_pl(image_id, box=(50.0, 50.0, 150.0, 150.0), pred_class=3)
        rd = assemble_dataset_only(labeled, [pl], 1, cfg)

        label_file = rd / "labels" / "train" / f"{image_id}.txt"
        lines = [l for l in label_file.read_text().splitlines() if l.strip()]
        # Original 1 GT box + 1 pseudo = 2 lines
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {lines}"
    print("PASS  test_pseudo_label_appended_to_labeled_image")


# ── Error handling ────────────────────────────────────────────────────────────

def test_unresolvable_pseudo_label_raises():
    """
    Pseudo-label with image_id not in labeled or unlabeled dirs
    must raise FileNotFoundError — not silently skip.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=2)
        cfg     = default_config(str(tmp / "out"))

        pl = make_pl("ghost_image_not_on_disk", box=(10.0, 10.0, 50.0, 50.0))
        try:
            assemble_dataset_only(labeled, [pl], 1, cfg)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "ghost_image_not_on_disk" in str(e)
    print("PASS  test_unresolvable_pseudo_label_raises")


# ── Filename collision handling ───────────────────────────────────────────────

def test_filename_collision_raises():
    """
    MVP uniqueness rule: labeled and unlabeled image IDs must be globally
    unique. Collision is detected EAGERLY at dataset resolution time —
    before any I/O — so it raises even before pseudo-labels are processed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=1)
        cfg     = default_config(str(tmp / "out"))

        fname    = labeled["image_list"][0]
        image_id = Path(fname).stem

        # Put the same filename in unlabeled_image_dir — collision in directories
        # Validation happens at resolve time, so even an empty pseudo_labels list
        # triggers the raise.
        unlab_dir = Path(labeled["unlabeled_image_dir"])
        make_image(unlab_dir / fname)

        try:
            assemble_dataset_only(labeled, [], 0, cfg)
            assert False, "Should have raised ValueError on filename collision"
        except ValueError as e:
            assert image_id in str(e), (
                f"ValueError message should mention the conflicting stem, "
                f"got: {e}")
    print("PASS  test_filename_collision_raises")


# ── String-path input ─────────────────────────────────────────────────────────

def test_string_path_labeled_data():
    """Passing a string path resolves images/train and labels/train."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp  = Path(tmp)
        base = tmp / "dataset"
        img_dir   = base / "images" / "train"
        label_dir = base / "labels" / "train"
        img_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        make_image(img_dir / "a.jpg")
        (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        cfg = default_config(str(tmp / "out"))
        # Pass string path instead of dict
        rd = assemble_dataset_only(str(base), [], 0, cfg)
        assert (rd / "images" / "train" / "a.jpg").exists()
        assert (rd / "dataset.yaml").exists()
    print("PASS  test_string_path_labeled_data")


# ── YOLO format tests ─────────────────────────────────────────────────────────

def test_yolo_line_format():
    trainer = YOLOTrainer(TrainingConfig())
    pl   = make_pl("img", box=(100.0, 100.0, 300.0, 300.0), pred_class=5)
    line = trainer._pl_to_yolo_line(pl, img_h=480, img_w=640)
    assert line is not None
    parts = line.split()
    assert len(parts) == 5
    cls_id = int(parts[0])
    cx, cy, bw, bh = [float(x) for x in parts[1:]]
    assert cls_id == 5
    assert abs(cx - 200.0/640) < 1e-4
    assert abs(cy - 200.0/480) < 1e-4
    assert abs(bw - 200.0/640) < 1e-4
    assert abs(bh - 200.0/480) < 1e-4
    print("PASS  test_yolo_line_format")


def test_degenerate_box_skipped():
    trainer = YOLOTrainer(TrainingConfig())
    assert trainer._pl_to_yolo_line(
        make_pl("img", box=(100.0, 100.0, 100.0, 100.0)), 480, 640) is None
    assert trainer._pl_to_yolo_line(
        make_pl("img", box=(200.0, 200.0, 100.0, 100.0)), 480, 640) is None
    print("PASS  test_degenerate_box_skipped")


def test_all_coords_normalised():
    trainer = YOLOTrainer(TrainingConfig())
    pl   = make_pl("img", box=(580.0, 460.0, 640.0, 480.0), pred_class=1)
    line = trainer._pl_to_yolo_line(pl, img_h=480, img_w=640)
    assert line is not None
    for val in [float(x) for x in line.split()[1:]]:
        assert 0.0 <= val <= 1.0, f"Coordinate {val} outside [0,1]"
    print("PASS  test_all_coords_normalised")


def test_multiple_pseudo_labels_per_image():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=1)
        cfg     = default_config(str(tmp / "out"))
        fname    = labeled["image_list"][0]
        image_id = Path(fname).stem
        pls = [
            make_pl(image_id, box_id=f"{image_id}_r1_{i}",
                    box=(50.0+i*100, 50.0, 150.0+i*100, 150.0),
                    pred_class=i)
            for i in range(3)
        ]
        rd = assemble_dataset_only(labeled, pls, 1, cfg)
        lines = [l for l in
                 (rd / "labels" / "train" / f"{image_id}.txt")
                 .read_text().splitlines() if l.strip()]
        assert len(lines) == 4  # 1 original + 3 pseudo
    print("PASS  test_multiple_pseudo_labels_per_image")


def test_round_directory_naming():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp)
        cfg     = default_config(str(tmp / "out"))
        for rid in [0, 3, 10]:
            rd = assemble_dataset_only(labeled, [], rid, cfg)
            assert rd.name == f"round_{rid:04d}"
    print("PASS  test_round_directory_naming")


def test_assembly_is_idempotent():
    """
    Running assembly twice for the same round with the same pseudo-labels
    must produce the same label files — no duplicated lines.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=1)
        cfg     = default_config(str(tmp / "out"))

        fname    = labeled["image_list"][0]
        image_id = Path(fname).stem
        pl = make_pl(image_id, box=(50.0, 50.0, 150.0, 150.0), pred_class=2)

        # Run assembly twice for round 1 with the same pseudo-label
        assemble_dataset_only(labeled, [pl], 1, cfg)
        assemble_dataset_only(labeled, [pl], 1, cfg)

        label_file = cfg.output_dir and (
            Path(cfg.output_dir) / "round_0001" / "labels" / "train"
            / f"{image_id}.txt"
        )
        # Resolve via Path
        label_path = Path(cfg.output_dir) / "round_0001" / "labels" / "train" / f"{image_id}.txt"
        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
        # 1 original GT + 1 pseudo = 2 lines, not 3 or 4
        assert len(lines) == 2, (
            f"Expected 2 lines after two runs (idempotent), got {len(lines)}: {lines}")
    print("PASS  test_assembly_is_idempotent")


def test_validation_error_does_not_delete_previous_assembly():
    """
    If a collision is detected, the previous round assembly must survive.
    Validate-before-delete ensures a validation error leaves prior work intact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled_clean = make_labeled_dataset(tmp, n_images=1, unlabeled_count=0)
        cfg = default_config(str(tmp / "out"))

        # Round 0: valid assembly — no unlabeled dir, no collision possible
        rd0 = assemble_dataset_only(labeled_clean, [], 0, cfg)
        assert rd0.exists(), "Round 0 assembly should exist"
        yaml_path = rd0 / "dataset.yaml"
        assert yaml_path.exists()

        # Now create a dataset with a collision to trigger ValueError on round 1
        labeled_collision = make_labeled_dataset(tmp / "coll", n_images=1,
                                                  unlabeled_count=1)
        # Force a collision: put labeled image stem into unlabeled dir
        fname    = labeled_collision["image_list"][0]
        unlab    = Path(labeled_collision["unlabeled_image_dir"])
        make_image(unlab / fname)

        try:
            assemble_dataset_only(labeled_collision, [], 1, cfg)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Round 0 directory must still exist untouched
        assert rd0.exists(), (
            "Round 0 assembly was deleted by a failed round 1 — "
            "validation must happen before any destructive I/O")
        assert yaml_path.exists(), "Round 0 dataset.yaml deleted unexpectedly"
    print("PASS  test_validation_error_does_not_delete_previous_assembly")


def test_missing_label_dir_does_not_delete_previous_assembly():
    """label_dir validation must happen before rmtree."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labeled = make_labeled_dataset(tmp, n_images=1)
        cfg     = default_config(str(tmp / "out"))
        rd0 = assemble_dataset_only(labeled, [], 0, cfg)
        assert rd0.exists()
        bad = dict(labeled)
        bad["label_dir"] = str(tmp / "nonexistent_labels")
        try:
            assemble_dataset_only(bad, [], 1, cfg)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
        assert rd0.exists(), "Round 0 deleted by failed round 1 with bad label_dir"
    print("PASS  test_missing_label_dir_does_not_delete_previous_assembly")

def test_same_dir_disjoint_lists_passes():
    """COCO mode: same dir, disjoint image_list/unlabeled_list -> no error."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shared = tmp / "images" / "train"
        labels = tmp / "labels" / "train"
        shared.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(4):
            make_image(shared / f"img_{i:04d}.jpg")
            (labels / f"img_{i:04d}.txt").write_text("0 0.5 0.5 0.3 0.3")
        cfg = default_config(str(tmp / "out"))
        data = {
            "image_dir":           str(shared),
            "label_dir":           str(labels),
            "image_list":          ["img_0000.jpg", "img_0001.jpg"],
            "unlabeled_image_dir": str(shared),
            "unlabeled_list":      ["img_0002.jpg", "img_0003.jpg"],
        }
        rd = assemble_dataset_only(data, [], 0, cfg)
        assert rd.exists()
    print("PASS  test_same_dir_disjoint_lists_passes")

def test_same_dir_overlapping_lists_raises():
    """COCO mode: overlapping image_list/unlabeled_list -> ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shared = tmp / "images" / "train"
        labels = tmp / "labels" / "train"
        shared.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(3):
            make_image(shared / f"img_{i:04d}.jpg")
            (labels / f"img_{i:04d}.txt").write_text("0 0.5 0.5 0.3 0.3")
        cfg = default_config(str(tmp / "out"))
        data = {
            "image_dir":           str(shared),
            "label_dir":           str(labels),
            "image_list":          ["img_0000.jpg", "img_0001.jpg"],
            "unlabeled_image_dir": str(shared),
            "unlabeled_list":      ["img_0001.jpg", "img_0002.jpg"],
        }
        try:
            assemble_dataset_only(data, [], 0, cfg)
            assert False, "Should raise ValueError on overlap"
        except ValueError:
            pass
    print("PASS  test_same_dir_overlapping_lists_raises")

# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_yaml_written,
        test_yaml_fields,
        test_dirs_created,
        test_labeled_images_and_labels_copied,
        test_pseudo_label_from_unlabeled_image_copied,
        test_pseudo_label_appended_to_labeled_image,
        test_unresolvable_pseudo_label_raises,
        test_filename_collision_raises,
        test_string_path_labeled_data,
        test_yolo_line_format,
        test_degenerate_box_skipped,
        test_all_coords_normalised,
        test_multiple_pseudo_labels_per_image,
        test_round_directory_naming,
        test_assembly_is_idempotent,
        test_validation_error_does_not_delete_previous_assembly,
        test_missing_label_dir_does_not_delete_previous_assembly,
        test_same_dir_disjoint_lists_passes,
        test_same_dir_overlapping_lists_raises,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    print(f"Phase 6B trainer tests: {passed} passed, {failed} failed")
    print("=" * 50)