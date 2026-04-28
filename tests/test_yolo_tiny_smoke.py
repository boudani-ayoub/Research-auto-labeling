"""
tests/test_yolo_tiny_smoke.py
==============================
Phase 6C — Real YOLO training integration smoke test.

Purpose:
  Prove the full chain works end-to-end with real Ultralytics training:
    assembled dataset.yaml
    → YOLOTrainer.train()
    → best.pt saved
    → checkpoint reloads cleanly
    → canonical_infer returns a CandidatePool

This is NOT an accuracy test. 1-2 epochs, tiny synthetic images.
Goal: no crashes, correct file outputs, correct return types.

Requires: GPU (RTX 4090 on server), yolov8n.pt, ultralytics installed.
Runtime: ~60-120 seconds.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import CandidatePool, PseudoLabel
from autolabel.inference.canonical_infer import canonical_infer
from autolabel.training.trainer import TrainingConfig, YOLOTrainer
from ultralytics import YOLO


DEVICE = "0" if torch.cuda.is_available() else "cpu"

# ── Tiny dataset builder ──────────────────────────────────────────────────────

def make_tiny_dataset(base_dir: Path,
                       n_images:   int = 8,
                       img_w:      int = 320,
                       img_h:      int = 320) -> dict:
    """
    Create a minimal YOLO-format dataset with synthetic images and labels.
    Each image is random noise (uint8 BGR). Each image has one random box.
    Returns a labeled_data dict with unlabeled_image_dir set.
    """
    import cv2

    img_dir   = base_dir / "images" / "train"
    label_dir = base_dir / "labels" / "train"
    unlab_dir = base_dir / "unlabeled"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    unlab_dir.mkdir(parents=True)

    rng        = np.random.default_rng(seed=7)
    image_list = []

    for i in range(n_images):
        fname = f"train_{i:04d}.jpg"
        img   = rng.integers(0, 256, (img_h, img_w, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / fname), img)

        # Random valid box
        x1 = rng.integers(0, img_w // 2)
        y1 = rng.integers(0, img_h // 2)
        x2 = rng.integers(img_w // 2, img_w)
        y2 = rng.integers(img_h // 2, img_h)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cls = int(rng.integers(0, 3))  # 3 classes

        with open(label_dir / f"train_{i:04d}.txt", "w") as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        image_list.append(fname)

    # Two unlabeled images for inference after training
    for i in range(2):
        fname = f"unlabeled_{i:04d}.jpg"
        img   = rng.integers(0, 256, (img_h, img_w, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(unlab_dir / fname), img)

    return {
        "image_dir":           str(img_dir),
        "label_dir":           str(label_dir),
        "image_list":          image_list,
        "unlabeled_image_dir": str(unlab_dir),
    }


def make_scores(pred_class: int, nc: int = 3) -> tuple:
    s = [0.0] * nc; s[pred_class] = 1.0
    return tuple(s)


def make_pl(image_id: str, box_id: str, box, pred_class: int) -> PseudoLabel:
    return PseudoLabel(
        image_id    = image_id,
        box_id      = box_id,
        round_id    = 1,
        box         = box,
        pred_class  = pred_class,
        class_scores= make_scores(pred_class, nc=3),
        confidence  = 0.75,
    )


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_trainer_produces_checkpoint():
    """
    YOLOTrainer.train() must:
      1. Assemble the YOLO dataset
      2. Run YOLO.train() for 1 epoch
      3. Return a path where best.pt exists
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp  = Path(tmp)
        data = make_tiny_dataset(tmp / "dataset", n_images=8)

        cfg = TrainingConfig(
            epochs     = 1,
            batch      = 4,
            device     = DEVICE,
            workers    = 0,
            pretrained = True,
            base_model = "yolov8n.pt",
            imgsz      = 320,
            output_dir = str(tmp / "outputs"),
            num_classes= 3,
            class_names= ["cat", "dog", "bird"],
        )

        trainer    = YOLOTrainer(cfg)
        checkpoint = trainer.train(
            labeled_data  = data,
            pseudo_labels = [],
            round_id      = 0,
        )

        assert checkpoint is not None, "trainer.train() returned None"
        assert Path(checkpoint).exists(), (
            f"Checkpoint path {checkpoint!r} does not exist")
        assert checkpoint.endswith(".pt"), (
            f"Expected .pt file, got: {checkpoint}")

        print(f"  Checkpoint: {checkpoint}")
    print("PASS  test_trainer_produces_checkpoint")


def test_checkpoint_reloads():
    """
    The checkpoint returned by trainer.train() must reload cleanly
    with ultralytics.YOLO() without errors.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp  = Path(tmp)
        data = make_tiny_dataset(tmp / "dataset", n_images=8)

        cfg = TrainingConfig(
            epochs=1, batch=4, device=DEVICE, workers=0,
            base_model="yolov8n.pt", imgsz=320,
            output_dir=str(tmp / "out"),
            num_classes=3, class_names=["cat", "dog", "bird"],
        )
        trainer    = YOLOTrainer(cfg)
        checkpoint = trainer.train(data, [], round_id=0)

        # Must reload without exception
        model = YOLO(checkpoint)
        assert model is not None

        print(f"  Reloaded: {checkpoint}")
    print("PASS  test_checkpoint_reloads")


def test_canonical_infer_after_training():
    """
    After training, canonical_infer() must return a valid CandidatePool
    for the unlabeled images. Checks return type, image_id keys, and
    that each detection is a valid PseudoLabel.
    """
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        tmp  = Path(tmp)
        data = make_tiny_dataset(tmp / "dataset", n_images=8)

        cfg = TrainingConfig(
            epochs=1, batch=4, device=DEVICE, workers=0,
            base_model="yolov8n.pt", imgsz=320,
            output_dir=str(tmp / "out"),
            num_classes=3, class_names=["cat", "dog", "bird"],
        )
        trainer    = YOLOTrainer(cfg)
        checkpoint = trainer.train(data, [], round_id=0)

        # Load checkpoint and run canonical inference
        model = YOLO(checkpoint)

        unlab_dir = Path(data["unlabeled_image_dir"])
        image_ids = []
        images    = []
        for p in sorted(unlab_dir.glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                image_ids.append(p.stem)
                images.append(img)

        assert len(images) > 0, "No unlabeled images found"

        pool = canonical_infer(
            model     = model,
            image_ids = image_ids,
            images    = images,
            round_id  = 1,
            conf      = 0.05,
            iou_nms   = 0.45,
            imgsz     = 320,
            batch_size= 2,
        )

        # Return type check
        assert isinstance(pool, dict), f"Expected dict, got {type(pool)}"

        # Every image_id must be in pool
        for iid in image_ids:
            assert iid in pool, f"image_id {iid!r} missing from pool"

        # Each detection must be a valid PseudoLabel
        for iid, pls in pool.items():
            assert isinstance(pls, list)
            for pl in pls:
                assert isinstance(pl, PseudoLabel)
                assert pl.image_id  == iid
                assert pl.round_id  == 1
                assert 0 <= pl.pred_class < 3
                assert 0.0 < pl.confidence <= 1.0
                assert len(pl.class_scores) == 3
                assert abs(sum(pl.class_scores) - 1.0) < 1e-4

        total_dets = sum(len(v) for v in pool.values())
        print(f"  Detections on {len(image_ids)} unlabeled images: {total_dets}")
    print("PASS  test_canonical_infer_after_training")


def test_second_round_with_pseudo_labels():
    """
    Full round 0 → round 1 chain:
      Round 0: train on labeled only
      Round 1: assemble labeled + pseudo-labels, train, get new checkpoint
    Verifies the complete trainer loop works twice.
    """
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        tmp  = Path(tmp)
        data = make_tiny_dataset(tmp / "dataset", n_images=8)

        cfg = TrainingConfig(
            epochs=1, batch=4, device=DEVICE, workers=0,
            base_model="yolov8n.pt", imgsz=320,
            output_dir=str(tmp / "out"),
            num_classes=3, class_names=["cat", "dog", "bird"],
        )
        trainer = YOLOTrainer(cfg)

        # Round 0 — labeled only
        ckpt0 = trainer.train(data, [], round_id=0)
        assert Path(ckpt0).exists()

        # Build 2 synthetic pseudo-labels from unlabeled images
        unlab_dir = Path(data["unlabeled_image_dir"])
        pseudo_labels = []
        for i, p in enumerate(sorted(unlab_dir.glob("*.jpg"))[:2]):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            pseudo_labels.append(make_pl(
                image_id  = p.stem,
                box_id    = f"{p.stem}_r1_{i:04d}",
                box       = (w*0.2, h*0.2, w*0.7, h*0.7),
                pred_class= i % 3,
            ))

        # Round 1 — labeled + pseudo-labels
        ckpt1 = trainer.train(data, pseudo_labels, round_id=1)
        assert Path(ckpt1).exists(), f"Round 1 checkpoint missing: {ckpt1}"
        assert ckpt0 != ckpt1 or Path(ckpt0).parent != Path(ckpt1).parent, (
            "Round 0 and Round 1 checkpoints should be in different directories")

        print(f"  Round 0: {ckpt0}")
        print(f"  Round 1: {ckpt1}")
    print("PASS  test_second_round_with_pseudo_labels")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 6C — Real YOLO training smoke test")
    print("Requires: GPU, yolov8n.pt, ~60-120s")
    print()

    tests = [
        test_trainer_produces_checkpoint,
        test_checkpoint_reloads,
        test_canonical_infer_after_training,
        test_second_round_with_pseudo_labels,
    ]

    passed = 0
    failed = 0
    for t in tests:
        print(f"Running {t.__name__} ...")
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
    print(f"Phase 6C smoke tests: {passed} passed, {failed} failed")
    print("=" * 50)