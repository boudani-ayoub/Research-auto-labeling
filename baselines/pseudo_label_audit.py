"""
baselines/pseudo_label_audit.py
================================
Pseudo-label audit for Baseline B planning.

Runs canonical inference using a trained checkpoint over the unlabeled
image pool and reports the distribution of detections BEFORE committing
to any training loop. This is inference-only — no training, no bank writes,
no round management.

Why run this first
------------------
Before implementing Baseline B (naive iterative pseudo-labeling), we need
to know:
  - How many pseudo-labels would be admitted at τ=0.50?
  - Which classes dominate?
  - Is the confidence distribution healthy or degenerate?
  - Will naive fixed-threshold B flood itself or starve minority classes?

If the audit shows saturation (>70% images pass) or severe class imbalance
(one class >70% of admitted boxes), that is NOT a reason to change the
baseline. It is evidence that confidence-only pseudo-labeling is uncontrolled,
which directly motivates method C's stability gate. Report it, then run B.

Output
------
  - Console summary table (per-class counts, confidence stats)
  - JSON audit file at --output_dir/audit.json
  - Optional: CSV of per-image stats at --output_dir/per_image_stats.csv

Usage
-----
  python baselines/pseudo_label_audit.py \\
      --checkpoint outputs/baseline_a/vehicles_fold0_10pct/training/round_0000/train/weights/best.pt \\
      --split_file  data/splits_vehicles/coco_vehicles_fold0_10pct.json \\
      --output_dir  outputs/baseline_b_audit \\
      --tau         0.50 \\
      --inference_conf 0.05 \\
      --device      0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_class_names(split: Dict) -> List[str]:
    """Read class names from split JSON (filtered schema) or fall back."""
    if "class_names" in split:
        return split["class_names"]
    # Legacy fallback: integer names
    return [str(i) for i in range(80)]


def _image_dir_from_split(split: Dict) -> Path:
    """Get the symlinked filtered image dir from a filtered split JSON."""
    if "image_dir_train" in split:
        return Path(split["image_dir_train"])
    # Legacy: coco_train_dir
    return Path(split["coco_train_dir"])


# ── Core audit logic ─────────────────────────────────────────────────────────

def run_audit(checkpoint:      str,
              split_file:       str,
              output_dir:       str,
              tau:              float = 0.50,
              inference_conf:   float = 0.05,
              imgsz:            int   = 640,
              batch:            int   = 32,
              device:           str   = "0",
              save_per_image:   bool  = True) -> Dict:
    """
    Run pseudo-label audit over unlabeled image pool.

    Two thresholds serve distinct purposes:
      inference_conf: loose gate passed to YOLO for NMS (collect all candidates)
      tau:            admission threshold (which candidates would be admitted)

    Both thresholds are applied independently. Do NOT raise inference_conf
    to tau — that discards the low-confidence distribution needed for the audit.

    Returns the audit result dict (also written to disk).
    """
    from ultralytics import YOLO

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load split
    with open(split_file) as f:
        split = json.load(f)

    class_names = _load_class_names(split)
    n_classes   = len(class_names)
    image_dir   = _image_dir_from_split(split)

    unlabeled_ids: List[int] = split["unlabeled_ids"]
    unlabeled_fnames = [f"{iid:012d}.jpg" for iid in unlabeled_ids]

    print()
    print("=" * 76)
    print("  Pseudo-label audit (inference-only)")
    print("=" * 76)
    print(f"  Checkpoint:     {checkpoint}")
    print(f"  Image dir:      {image_dir}")
    print(f"  Unlabeled pool: {len(unlabeled_ids):,} images")
    print(f"  inference_conf: {inference_conf}  (loose — collects all candidates)")
    print(f"  admission_tau:  {tau}  (threshold for would-be admission)")
    print(f"  Output:         {output_dir_path}")
    print()

    # ── Load model ───────────────────────────────────────────────────────────
    model = YOLO(checkpoint)

    # ── Accumulators ─────────────────────────────────────────────────────────
    # All detections (before τ)
    total_det_count:    int = 0
    # Admitted detections (conf ≥ τ)
    admitted_count:     int = 0
    # Per-class counters
    det_per_class:     Dict[int, int] = defaultdict(int)
    admitted_per_class: Dict[int, int] = defaultdict(int)
    # Confidence accumulators per class (for mean/median)
    conf_per_class:    Dict[int, List[float]] = defaultdict(list)
    admitted_conf_per_class: Dict[int, List[float]] = defaultdict(list)
    # Image-level
    images_with_any_det:     int = 0
    images_with_admitted_det: int = 0
    # Per-image stats (for CSV / detailed analysis)
    per_image_rows: List[Dict] = []

    n_total = len(unlabeled_fnames)
    t_start = time.time()

    # ── Inference in batches ─────────────────────────────────────────────────
    print(f"Running inference on {n_total:,} images (batch={batch}) ...")
    for batch_start in range(0, n_total, batch):
        batch_fnames = unlabeled_fnames[batch_start: batch_start + batch]
        batch_paths  = [str(image_dir / fn) for fn in batch_fnames]

        # Run model
        results_list = model.predict(
            source  = batch_paths,
            imgsz   = imgsz,
            conf    = inference_conf,
            iou     = 0.45,
            device  = device,
            verbose = False,
        )

        for fname, results in zip(batch_fnames, results_list):
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                per_image_rows.append({
                    "fname":         fname,
                    "n_det":         0,
                    "n_admitted":    0,
                    "max_conf":      None,
                    "admitted_classes": [],
                })
                continue

            confs   = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()

            n_det = len(confs)
            total_det_count += n_det
            images_with_any_det += 1

            admitted_in_image = []
            for conf_val, cls_val in zip(confs, classes):
                cid = int(cls_val)
                det_per_class[cid]  += 1
                conf_per_class[cid].append(conf_val)

                if conf_val >= tau:
                    admitted_count            += 1
                    admitted_per_class[cid]   += 1
                    admitted_conf_per_class[cid].append(conf_val)
                    admitted_in_image.append(cid)

            if admitted_in_image:
                images_with_admitted_det += 1

            per_image_rows.append({
                "fname":            fname,
                "n_det":            n_det,
                "n_admitted":       len(admitted_in_image),
                "max_conf":         max(confs) if confs else None,
                "admitted_classes": admitted_in_image,
            })

        # Progress
        done = min(batch_start + batch, n_total)
        elapsed = time.time() - t_start
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (n_total - done) / speed if speed > 0 else 0
        print(f"  {done:>6}/{n_total}  "
              f"admitted so far: {admitted_count:,}  "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
              end="\r", flush=True)

    elapsed_total = time.time() - t_start
    print()
    print(f"  Done in {elapsed_total:.1f}s ({n_total/elapsed_total:.1f} img/s)")
    print()

    # ── Compute per-class stats ───────────────────────────────────────────────
    def _median(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        mid = len(s) // 2
        return (s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid]

    def _mean(vals: List[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    per_class_stats = {}
    for cid in range(n_classes):
        name = class_names[cid]
        det_confs = conf_per_class.get(cid, [])
        adm_confs = admitted_conf_per_class.get(cid, [])
        per_class_stats[name] = {
            "class_id":              cid,
            "det_count":             det_per_class.get(cid, 0),
            "admitted_count":        admitted_per_class.get(cid, 0),
            "det_pct_of_total":      det_per_class.get(cid, 0) / max(total_det_count, 1) * 100,
            "admitted_pct_of_total": admitted_per_class.get(cid, 0) / max(admitted_count, 1) * 100,
            "mean_conf_all_dets":    _mean(det_confs),
            "median_conf_all_dets":  _median(det_confs),
            "mean_conf_admitted":    _mean(adm_confs),
            "median_conf_admitted":  _median(adm_confs),
        }

    # ── Saturation and imbalance flags ────────────────────────────────────────
    saturation_pct = images_with_admitted_det / max(n_total, 1) * 100
    top_class_pct  = max(
        (s["admitted_pct_of_total"] for s in per_class_stats.values()), default=0
    )
    top_class_name = max(
        per_class_stats,
        key=lambda k: per_class_stats[k]["admitted_count"],
        default="?"
    )

    saturation_flag = saturation_pct > 70
    imbalance_flag  = top_class_pct  > 60

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 76)
    print("  Per-class results")
    print("=" * 76)
    print(f"  {'class':<14}  {'det':>8}  {'adm':>8}  "
          f"{'det%':>6}  {'adm%':>6}  "
          f"{'mean_conf_all':>14}  {'mean_conf_adm':>14}")
    print("  " + "-" * 74)
    for name, s in per_class_stats.items():
        print(f"  {name:<14}  "
              f"{s['det_count']:>8,}  "
              f"{s['admitted_count']:>8,}  "
              f"{s['det_pct_of_total']:>5.1f}%  "
              f"{s['admitted_pct_of_total']:>5.1f}%  "
              f"{(s['mean_conf_all_dets'] or 0):>14.4f}  "
              f"{(s['mean_conf_admitted'] or 0):>14.4f}")

    print()
    print("=" * 76)
    print("  Summary")
    print("=" * 76)
    print(f"  unlabeled images:              {n_total:,}")
    print(f"  images with any detection:     {images_with_any_det:,}  "
          f"({images_with_any_det/max(n_total,1)*100:.1f}%)")
    print(f"  images with admitted det:      {images_with_admitted_det:,}  "
          f"({saturation_pct:.1f}%)")
    print(f"  total detections (before τ):   {total_det_count:,}")
    print(f"  admitted pseudo-labels (≥{tau}): {admitted_count:,}")
    print(f"  top class by admitted count:   {top_class_name} "
          f"({top_class_pct:.1f}% of admitted)")
    print()
    if saturation_flag:
        print(f"  ⚠ SATURATION: {saturation_pct:.1f}% of images pass τ={tau}. "
              f"Run fixed-threshold B anyway — this shows confidence-only "
              f"admission is uncontrolled, which motivates method C.")
    if imbalance_flag:
        print(f"  ⚠ IMBALANCE: {top_class_name} dominates ({top_class_pct:.1f}% "
              f"of admitted). This is a documented failure mode of confidence-only "
              f"pseudo-labeling. Report it; it motivates the stability gate.")
    if not saturation_flag and not imbalance_flag:
        print(f"  ✓ No saturation or severe imbalance detected. "
              f"Pure fixed-threshold B is appropriate.")

    # ── Save audit JSON ───────────────────────────────────────────────────────
    audit = {
        "checkpoint":               str(checkpoint),
        "split_file":               str(split_file),
        "inference_conf":           inference_conf,
        "admission_tau":            tau,
        "n_unlabeled":              n_total,
        "images_with_any_det":      images_with_any_det,
        "images_with_admitted_det": images_with_admitted_det,
        "saturation_pct":           saturation_pct,
        "total_det_before_tau":     total_det_count,
        "admitted_count":           admitted_count,
        "top_class_name":           top_class_name,
        "top_class_pct":            top_class_pct,
        "saturation_flag":          saturation_flag,
        "imbalance_flag":           imbalance_flag,
        "per_class":                per_class_stats,
        "created_at":               datetime.now(timezone.utc).isoformat(),
    }
    audit_path = output_dir_path / "audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print()
    print(f"  Audit saved to: {audit_path}")

    # ── Optional: per-image CSV ───────────────────────────────────────────────
    if save_per_image:
        csv_path = output_dir_path / "per_image_stats.csv"
        with open(csv_path, "w") as f:
            f.write("fname,n_det,n_admitted,max_conf\n")
            for row in per_image_rows:
                f.write(f"{row['fname']},{row['n_det']},"
                        f"{row['n_admitted']},"
                        f"{row['max_conf'] if row['max_conf'] is not None else ''}\n")
        print(f"  Per-image CSV:  {csv_path}")

    print()
    return audit


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pseudo-label audit — inference-only, no training"
    )
    parser.add_argument("--checkpoint",     required=True,
        help="Path to trained .pt checkpoint (use Baseline A best.pt)")
    parser.add_argument("--split_file",     required=True,
        help="Split JSON (filtered schema — needs unlabeled_ids + image_dir_train)")
    parser.add_argument("--output_dir",     required=True,
        help="Directory to save audit.json and per_image_stats.csv")
    parser.add_argument("--tau",            type=float, default=0.50,
        help="Admission threshold to audit (default: 0.50)")
    parser.add_argument("--inference_conf", type=float, default=0.05,
        help="Inference conf for YOLO NMS (default: 0.05 — loose, collects all "
             "candidates; do NOT raise to tau, that discards distribution info)")
    parser.add_argument("--imgsz",          type=int,   default=640)
    parser.add_argument("--batch",          type=int,   default=32)
    parser.add_argument("--device",         default="0" if torch.cuda.is_available()
                                                     else "cpu")
    parser.add_argument("--no_per_image",   action="store_true",
        help="Skip saving per-image CSV (saves time on very large pools)")
    args = parser.parse_args()

    run_audit(
        checkpoint     = args.checkpoint,
        split_file     = args.split_file,
        output_dir     = args.output_dir,
        tau            = args.tau,
        inference_conf = args.inference_conf,
        imgsz          = args.imgsz,
        batch          = args.batch,
        device         = args.device,
        save_per_image = not args.no_per_image,
    )