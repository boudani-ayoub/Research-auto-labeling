"""
baselines/baseline_a.py
=======================
Baseline A: Supervised-only training on the labeled set.

No pseudo-labels. No iterative loop. No unlabeled data.
Train YOLOv8m on the labeled split once, evaluate on val2017.

This is the primary comparison point for the paper.
The claim "our method improves over supervised-only" requires this number.

Protocol:
  1. Load a COCO split JSON (from data/make_splits.py)
  2. Assemble YOLO dataset from labeled images only
  3. Train YOLOv8m for config.epochs
  4. Evaluate on val2017
  5. Save checkpoint + eval result to JSON

Usage:
  python baselines/baseline_a.py \\
      --split_file data/splits/coco_fold0_10pct.json \\
      --coco_label_dir ~/research/coco/labels/train2017 \\
      --val_image_dir ~/research/coco/images/val2017 \\
      --val_label_dir ~/research/coco/labels/val2017 \\
      --output_dir outputs/baseline_a/fold0_10pct \\
      --epochs 100 \\
      --device 0

  # All 5 folds at 10%:
  for fold in 0 1 2 3 4; do
      python baselines/baseline_a.py \\
          --split_file data/splits/coco_fold${fold}_10pct.json \\
          --coco_label_dir ~/research/coco/labels/train2017 \\
          --val_image_dir ~/research/coco/images/val2017 \\
          --val_label_dir ~/research/coco/labels/val2017 \\
          --output_dir outputs/baseline_a/fold${fold}_10pct \\
          --epochs 100
  done
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from autolabel.training.trainer import TrainingConfig, YOLOTrainer
from data.make_splits import load_split, split_to_labeled_data
from eval.evaluate import evaluate_checkpoint, save_results


def run_baseline_a(split_file:      str,
                   coco_label_dir:  str,
                   val_image_dir:   str,
                   val_label_dir:   str,
                   output_dir:      str,
                   epochs:          int   = 100,
                   batch:           int   = 16,
                   imgsz:           int   = 640,
                   device:          str   = "0",
                   base_model:      str   = "yolov8m.pt",
                   workers:         int   = 8) -> dict:
    """
    Run the supervised-only baseline for one split.

    Returns a summary dict with checkpoint path and mAP results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load split ────────────────────────────────────────────────────────────
    print(f"Loading split: {split_file}")
    split = load_split(split_file)
    print(f"  ratio={split['ratio']:.0%}  fold={split['fold']}  "
          f"n_labeled={split['n_labeled']:,}")

    labeled_data = split_to_labeled_data(split, label_dir=coco_label_dir)

    # ── Train ─────────────────────────────────────────────────────────────────
    cfg = TrainingConfig(
        epochs      = epochs,
        batch       = batch,
        imgsz       = imgsz,
        device      = device,
        workers     = workers,
        pretrained  = True,
        base_model  = base_model,
        output_dir  = str(output_dir / "training"),
        num_classes = 80,
    )

    print(f"Training {base_model} on {split['n_labeled']:,} labeled images "
          f"for {epochs} epochs ...")
    trainer    = YOLOTrainer(cfg)
    checkpoint = trainer.train(
        labeled_data  = labeled_data,
        pseudo_labels = [],   # Baseline A: no pseudo-labels
        round_id      = 0,
    )
    print(f"  Checkpoint: {checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"Evaluating on val2017 ...")
    result = evaluate_checkpoint(
        checkpoint    = checkpoint,
        val_image_dir = val_image_dir,
        val_label_dir = val_label_dir,
        num_classes   = 80,
        imgsz         = imgsz,
        batch         = batch,
        device        = device,
        round_id      = 0,
    )
    print(f"  mAP50:    {result.map50:.4f}")
    print(f"  mAP50-95: {result.map50_95:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    eval_file = output_dir / "eval_result.json"
    save_results(
        results     = [result],
        output_file = str(eval_file),
        metadata    = {
            "baseline":   "A_supervised_only",
            "split_file": str(split_file),
            "ratio":      split["ratio"],
            "fold":       split["fold"],
            "n_labeled":  split["n_labeled"],
            "base_model": base_model,
            "epochs":     epochs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    summary = {
        "baseline":   "A",
        "ratio":      split["ratio"],
        "fold":       split["fold"],
        "n_labeled":  split["n_labeled"],
        "checkpoint": checkpoint,
        "map50":      result.map50,
        "map50_95":   result.map50_95,
        "eval_file":  str(eval_file),
    }

    # Also save summary JSON for easy loading by metrics_table
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline A: supervised-only training on labeled split"
    )
    parser.add_argument("--split_file",     required=True,
        help="Path to split JSON from data/make_splits.py")
    parser.add_argument("--coco_label_dir", required=True,
        help="Path to YOLO-format train2017 label directory")
    parser.add_argument("--val_image_dir",  required=True,
        help="Path to val2017 images directory")
    parser.add_argument("--val_label_dir",  required=True,
        help="Path to YOLO-format val2017 labels directory")
    parser.add_argument("--output_dir",     required=True,
        help="Directory to save checkpoints and results")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--device",   default="0" if torch.cuda.is_available()
                                                else "cpu")
    parser.add_argument("--base_model", default="yolov8m.pt")
    parser.add_argument("--workers",  type=int,   default=8)

    args = parser.parse_args()

    summary = run_baseline_a(
        split_file     = args.split_file,
        coco_label_dir = args.coco_label_dir,
        val_image_dir  = args.val_image_dir,
        val_label_dir  = args.val_label_dir,
        output_dir     = args.output_dir,
        epochs         = args.epochs,
        batch          = args.batch,
        imgsz          = args.imgsz,
        device         = args.device,
        base_model     = args.base_model,
        workers        = args.workers,
    )

    print()
    print("=" * 50)
    print("Baseline A complete")
    print(f"  mAP50:    {summary['map50']:.4f}")
    print(f"  mAP50-95: {summary['map50_95']:.4f}")
    print("=" * 50)