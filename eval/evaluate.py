"""
eval/evaluate.py
================
Evaluation module for the stability-gated iterative auto-labeling pipeline.

Runs YOLOv8 validation on COCO val2017 and returns mAP metrics.

Architecture notes:
  - Evaluation is COMPLETELY INDEPENDENT of the outer loop.
  - No labeled validation set is used at any point in the pipeline loop.
    Stopping is unsupervised. Evaluation is post-hoc only.
  - Evaluation uses the standard COCO val2017 split (5,000 images).
  - Metrics reported: mAP50, mAP50-95 (COCO standard), and per-class AP.

Usage:
  # Evaluate a single checkpoint
  python eval/evaluate.py \\
      --checkpoint outputs/round_0003/train/weights/best.pt \\
      --val_image_dir ~/research/coco/images/val2017 \\
      --val_label_dir ~/research/coco/labels/val2017 \\
      --num_classes 80 \\
      --output_file outputs/eval_round3.json

  # Evaluate all rounds from a bank
  python eval/evaluate.py \\
      --bank_dir outputs/bank \\
      --val_image_dir ~/research/coco/images/val2017 \\
      --val_label_dir ~/research/coco/labels/val2017 \\
      --num_classes 80 \\
      --output_file outputs/eval_all_rounds.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path regardless of how the script is invoked
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class EvalResult:
    """
    Evaluation results for one checkpoint.

    map50:      mAP at IoU=0.50 (PASCAL VOC style)
    map50_95:   mAP at IoU=0.50:0.95 (COCO standard)
    per_class:  dict mapping class_name → AP50 (if available)
    checkpoint: path to the evaluated checkpoint
    round_id:   pipeline round this checkpoint came from (-1 if unknown)
    n_images:   number of val images evaluated
    """
    map50:      float
    map50_95:   float
    per_class:  Dict[str, float]
    checkpoint: str
    round_id:   int   = -1
    n_images:   int   = 0
    speed_ms:   float = 0.0   # inference ms per image

    def to_dict(self) -> dict:
        return {
            "map50":      self.map50,
            "map50_95":   self.map50_95,
            "per_class":  self.per_class,
            "checkpoint": self.checkpoint,
            "round_id":   self.round_id,
            "n_images":   self.n_images,
            "speed_ms":   self.speed_ms,
        }


# ── Dataset.yaml writer ───────────────────────────────────────────────────────

def _write_val_yaml(val_image_dir: str,
                    val_label_dir: str,
                    num_classes:   int,
                    class_names:   Optional[List[str]],
                    tmp_dir:       Path) -> Path:
    """
    Write a minimal dataset.yaml for validation only.

    Creates a YOLO-standard directory structure using symlinks:
        tmp_dir/images/val2017 → val_image_dir
        tmp_dir/labels/val2017 → val_label_dir

    This ensures val_label_dir is actually respected by YOLO's label
    inference logic, which derives label paths from image paths.
    """
    import yaml

    # Build YOLO-standard parallel structure
    tmp_images = tmp_dir / "images" / "val2017"
    tmp_labels = tmp_dir / "labels" / "val2017"
    tmp_images.parent.mkdir(parents=True, exist_ok=True)
    tmp_labels.parent.mkdir(parents=True, exist_ok=True)
    tmp_images.symlink_to(Path(val_image_dir).resolve(),
                           target_is_directory=True)
    tmp_labels.symlink_to(Path(val_label_dir).resolve(),
                           target_is_directory=True)

    names = class_names or [str(i) for i in range(num_classes)]
    data  = {
        "path":  str(tmp_dir),
        "train": "images/val2017",   # placeholder — not used for val
        "val":   "images/val2017",
        "nc":    num_classes,
        "names": names,
    }
    yaml_path = tmp_dir / "val_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint:    str,
                         val_image_dir: str,
                         val_label_dir: str,
                         num_classes:   int   = 80,
                         class_names:   Optional[List[str]] = None,
                         imgsz:         int   = 640,
                         batch:         int   = 16,
                         device:        str   = "0",
                         round_id:      int   = -1,
                         conf:          float = 0.001,
                         iou:           float = 0.65) -> EvalResult:
    """
    Run YOLO validation on val2017 and return an EvalResult.

    Args:
        checkpoint:    path to .pt checkpoint file
        val_image_dir: path to val2017 images directory
        val_label_dir: path to YOLO-format val2017 labels directory
        num_classes:   number of object classes (80 for COCO, 5 for
                        filtered vehicles)
        class_names:   list of class names (None → integer indices).
                        Strongly recommended for the filtered experiment
                        so per-class AP is labeled "bicycle"/"car"/...
        imgsz:         validation image size (default 640)
        batch:         validation batch size
        device:        CUDA device index or 'cpu'
        round_id:      pipeline round for bookkeeping (-1 if unknown)
        conf:          confidence threshold for NMS (low for mAP eval)
        iou:           IoU threshold for NMS

    Returns:
        EvalResult with mAP50, mAP50-95, per-class AP

    Raises:
        ValueError if class_names length does not match num_classes
    """
    import torch
    from ultralytics import YOLO

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but "
            f"num_classes={num_classes}. They must match."
        )

    checkpoint = str(Path(checkpoint).resolve())
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not Path(val_image_dir).exists():
        raise FileNotFoundError(f"val_image_dir not found: {val_image_dir}")

    if not Path(val_label_dir).exists():
        raise FileNotFoundError(
            f"val_label_dir not found: {val_label_dir}\n"
            f"Run: python data/make_splits.py --convert_val_labels ..."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir  = Path(tmp)

        yaml_path = _write_val_yaml(
            val_image_dir = val_image_dir,
            val_label_dir = val_label_dir,
            num_classes   = num_classes,
            class_names   = class_names,
            tmp_dir       = tmp_dir,
        )

        model   = YOLO(checkpoint)
        results = model.val(
            data    = str(yaml_path),
            imgsz   = imgsz,
            batch   = batch,
            device  = device,
            conf    = conf,
            iou     = iou,
            verbose = False,
            save_json = False,
        )

        # Extract metrics
        map50    = float(results.box.map50)    if hasattr(results.box, 'map50')    else 0.0
        map50_95 = float(results.box.map)      if hasattr(results.box, 'map')      else 0.0
        speed    = float(results.speed.get('inference', 0.0)) if hasattr(results, 'speed') else 0.0

        # Per-class AP50
        per_class: Dict[str, float] = {}
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
            names = class_names or [str(i) for i in range(num_classes)]
            for idx, ap in zip(results.box.ap_class_index,
                               results.box.ap50):
                cls_name = names[int(idx)] if int(idx) < len(names) else str(int(idx))
                per_class[cls_name] = float(ap)

        # Count actual val images rather than hardcoding 5000
        n_images = len(list(Path(val_image_dir).glob("*.jpg")))

        return EvalResult(
            map50      = map50,
            map50_95   = map50_95,
            per_class  = per_class,
            checkpoint = checkpoint,
            round_id   = round_id,
            n_images   = n_images,
            speed_ms   = speed,
        )


# ── Multi-round evaluation ────────────────────────────────────────────────────

def evaluate_all_rounds(bank_dir:      str,
                         val_image_dir: str,
                         val_label_dir: str,
                         num_classes:   int   = 80,
                         class_names:   Optional[List[str]] = None,
                         imgsz:         int   = 640,
                         batch:         int   = 16,
                         device:        str   = "0") -> List[EvalResult]:
    """
    Evaluate all rounds from a bank directory.
    Reads RoundMetadata to find checkpoints, evaluates each in order.

    Returns list of EvalResult sorted by round_id.
    """
    from autolabel.bank.bank import PseudoLabelBank

    bank    = PseudoLabelBank.load_or_create(bank_dir)
    results = []

    for meta in bank.all_metadata():
        ckpt = meta.model_checkpoint
        if not ckpt or not Path(ckpt).exists():
            print(f"  Round {meta.round_id}: checkpoint missing ({ckpt}) — skipping")
            continue

        print(f"  Round {meta.round_id}: evaluating {Path(ckpt).name} ...")
        result = evaluate_checkpoint(
            checkpoint    = ckpt,
            val_image_dir = val_image_dir,
            val_label_dir = val_label_dir,
            num_classes   = num_classes,
            class_names   = class_names,
            imgsz         = imgsz,
            batch         = batch,
            device        = device,
            round_id      = meta.round_id,
        )
        print(f"    mAP50={result.map50:.4f}  mAP50-95={result.map50_95:.4f}")
        results.append(result)

    return sorted(results, key=lambda r: r.round_id)


# ── Results I/O ───────────────────────────────────────────────────────────────

def save_results(results: List[EvalResult],
                 output_file: str,
                 metadata: Optional[dict] = None) -> None:
    """Save evaluation results to a JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata":   metadata or {},
        "results":    [r.to_dict() for r in results],
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} result(s) to {output_file}")


def load_results(results_file: str) -> List[EvalResult]:
    """Load evaluation results from a JSON file."""
    with open(results_file, "r") as f:
        data = json.load(f)
    results = []
    for d in data["results"]:
        results.append(EvalResult(
            map50      = d["map50"],
            map50_95   = d["map50_95"],
            per_class  = d.get("per_class", {}),
            checkpoint = d["checkpoint"],
            round_id   = d.get("round_id", -1),
            n_images   = d.get("n_images", 0),
            speed_ms   = d.get("speed_ms", 0.0),
        ))
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 checkpoint(s) on COCO val2017"
    )

    # Input — either a single checkpoint or a bank directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint",
        help="Path to a single .pt checkpoint to evaluate")
    group.add_argument("--bank_dir",
        help="Bank directory — evaluates all rounds")

    # Val dataset
    parser.add_argument("--val_image_dir", required=True,
        help="Path to val2017 images directory")
    parser.add_argument("--val_label_dir", required=True,
        help="Path to YOLO-format val2017 labels directory")

    # Model/eval settings
    parser.add_argument("--num_classes", type=int, default=80)
    parser.add_argument("--class_names", default=None,
        help="Comma-separated class names matching num_classes "
             "(e.g. 'bicycle,car,motorcycle,bus,truck'). "
             "Required for filtered-class experiments to label per-class AP "
             "with names instead of integer indices.")
    parser.add_argument("--imgsz",       type=int, default=640)
    parser.add_argument("--batch",       type=int, default=16)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--round_id",    type=int, default=-1,
        help="Round ID for bookkeeping (single checkpoint mode only)")

    # Output
    parser.add_argument("--output_file", default=None,
        help="Path to save JSON results (optional)")

    args = parser.parse_args()

    # Parse class_names CSV
    class_names_list: Optional[List[str]] = None
    if args.class_names:
        class_names_list = [n.strip() for n in args.class_names.split(",")
                            if n.strip()]
        if len(class_names_list) != args.num_classes:
            parser.error(
                f"--class_names has {len(class_names_list)} entries but "
                f"--num_classes={args.num_classes}. They must match."
            )

    if args.checkpoint:
        print(f"Evaluating {args.checkpoint} ...")
        result = evaluate_checkpoint(
            checkpoint    = args.checkpoint,
            val_image_dir = args.val_image_dir,
            val_label_dir = args.val_label_dir,
            num_classes   = args.num_classes,
            class_names   = class_names_list,
            imgsz         = args.imgsz,
            batch         = args.batch,
            device        = args.device,
            round_id      = args.round_id,
        )
        print(f"\nResults:")
        print(f"  mAP50:    {result.map50:.4f}")
        print(f"  mAP50-95: {result.map50_95:.4f}")
        print(f"  Speed:    {result.speed_ms:.1f} ms/image")
        if result.per_class:
            print(f"  Per-class AP50:")
            for name, ap in result.per_class.items():
                print(f"    {name}: {ap:.4f}")
        results = [result]

    else:
        print(f"Evaluating all rounds from bank: {args.bank_dir}")
        results = evaluate_all_rounds(
            bank_dir      = args.bank_dir,
            val_image_dir = args.val_image_dir,
            val_label_dir = args.val_label_dir,
            num_classes   = args.num_classes,
            class_names   = class_names_list,
            imgsz         = args.imgsz,
            batch         = args.batch,
            device        = args.device,
        )
        print(f"\nSummary ({len(results)} rounds):")
        print(f"  {'Round':<8} {'mAP50':<10} {'mAP50-95':<12}")
        print(f"  {'-'*30}")
        for r in results:
            print(f"  {r.round_id:<8} {r.map50:<10.4f} {r.map50_95:<12.4f}")

    if args.output_file:
        save_results(results, args.output_file)