"""
baselines/baseline_b.py
=======================
Baseline B: Naive confidence-only iterative pseudo-labeling.

No stability scoring. No jitter. No matching. No class balancing.
Just: train → pseudo-label by confidence → retrain → repeat.

This is the "dumb iterative" baseline. It exists to show the failure mode
that motivates Method C (stability-gated admission). The audit showed that
τ=0.50 admits 13,188/17,783 images and 30,597 pseudo boxes at Round 1,
dominated by car (53.4%). This uncontrolled flooding is what we measure.

Design decisions (all locked):
  τ_conf:         0.50        fixed admission threshold, no per-class logic
  inference_conf: 0.05        loose gate for inference — collect all candidates
  rounds:         3           iterative rounds (after Round 0 = Baseline A)
  warm_start:     previous best.pt for each round
  pseudo-labels:  current-round only, not accumulated across rounds
  training:       100 epochs, patience=20, AdamW, lr0=0.001
  class balance:  none
  progressive:    none

Round protocol:
  Round 0  =  Baseline A checkpoint (supervised-only, provided via --init_checkpoint)
  Round 1  =  infer on unlabeled with Round-0 model → admit τ=0.50 → retrain
  Round 2  =  infer on unlabeled with Round-1 model → admit τ=0.50 → retrain
  Round 3  =  infer on unlabeled with Round-2 model → admit τ=0.50 → retrain

Each round:
  - Runs canonical inference at conf=0.05 on all unlabeled images
  - Admits detections with conf ≥ 0.50 as pseudo-labels
  - Trains from the previous round's best.pt on (labeled_train + admitted PLs)
  - Evaluates on filtered val2017
  - Logs per-class pseudo-label stats (same format as audit script)

Usage:
  python baselines/baseline_b.py \\
      --split_file       data/splits_vehicles/coco_vehicles_fold0_10pct.json \\
      --init_checkpoint  outputs/baseline_a/vehicles_fold0_10pct/training/round_0000/train/weights/best.pt \\
      --output_dir       outputs/baseline_b/vehicles_fold0_10pct \\
      --rounds           3 \\
      --epochs           100 \\
      --patience         20 \\
      --device           0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from autolabel.bank.schemas import PseudoLabel
from autolabel.training.trainer import TrainingConfig, YOLOTrainer
from data.make_splits import load_split, filtered_split_to_labeled_data
from eval.evaluate import evaluate_checkpoint, save_results


# ── Pseudo-label generation (confidence-only, no stability) ──────────────────

def generate_pseudo_labels(checkpoint:     str,
                            image_dir:      Path,
                            unlabeled_ids:  List[int],
                            inference_conf: float,
                            tau_conf:       float,
                            imgsz:          int,
                            batch:          int,
                            device:         str,
                            round_id:       int,
                            class_names:    List[str]) -> Tuple[List[PseudoLabel],
                                                                 Dict[str, object]]:
    """
    Run canonical inference on unlabeled images and admit detections with
    conf ≥ tau_conf as pseudo-labels.

    Two thresholds:
      inference_conf: passed to model.predict() NMS — collects all candidates
      tau_conf:       admission filter — only kept detections become PseudoLabels

    Returns:
        pseudo_labels:  list of PseudoLabel objects for admitted detections
        audit_stats:    per-class counts / confidence stats (same schema as
                        pseudo_label_audit.py) for per-round logging
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    n_classes = len(class_names)

    # Accumulators
    admitted:            List[PseudoLabel]         = []
    det_per_class:       Dict[int, int]            = defaultdict(int)
    admitted_per_class:  Dict[int, int]            = defaultdict(int)
    conf_per_class:      Dict[int, List[float]]    = defaultdict(list)
    adm_conf_per_class:  Dict[int, List[float]]    = defaultdict(list)
    images_with_any_det:     int = 0
    images_with_admitted:    int = 0
    total_det:               int = 0

    unlabeled_fnames = [f"{iid:012d}.jpg" for iid in unlabeled_ids]
    n_total = len(unlabeled_fnames)

    print(f"  Generating pseudo-labels for round {round_id} ...")
    print(f"    inference_conf={inference_conf}, τ={tau_conf}, "
          f"{n_total:,} unlabeled images ...")

    det_index = 0  # global detection index within this round for box_id

    for batch_start in range(0, n_total, batch):
        batch_fnames = unlabeled_fnames[batch_start: batch_start + batch]
        batch_paths  = [str(image_dir / fn) for fn in batch_fnames]

        results_list = model.predict(
            source  = batch_paths,
            imgsz   = imgsz,
            conf    = inference_conf,
            iou     = 0.45,
            device  = device,
            verbose = False,
        )

        for fname, results in zip(batch_fnames, results_list):
            image_id = Path(fname).stem   # "000000000100"
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                continue

            confs    = boxes.conf.cpu().tolist()
            classes  = boxes.cls.cpu().tolist()
            xyxy_all = boxes.xyxy.cpu().tolist()

            has_any_det = True
            images_with_any_det += 1
            has_admitted = False

            for conf_val, cls_val, xyxy in zip(confs, classes, xyxy_all):
                cid = int(cls_val)
                total_det += 1
                det_per_class[cid]  += 1
                conf_per_class[cid].append(conf_val)

                if conf_val >= tau_conf:
                    admitted_per_class[cid] += 1
                    adm_conf_per_class[cid].append(conf_val)
                    has_admitted = True

                    # Build a minimal PseudoLabel.
                    # class_scores: one-hot at pred_class (no softmax available
                    # in Baseline B — this is the naive baseline, we don't
                    # extract full class scores from the head).
                    # confidence: the post-NMS detection score.
                    # box: (x1, y1, x2, y2) in absolute pixels.
                    scores = [0.0] * n_classes
                    scores[cid] = 1.0   # one-hot: naive B doesn't have softmax

                    pl = PseudoLabel(
                        image_id     = image_id,
                        box_id       = f"{image_id}_r{round_id}_{det_index:05d}",
                        round_id     = round_id,
                        box          = tuple(xyxy),
                        pred_class   = cid,
                        class_scores = tuple(scores),
                        confidence   = conf_val,
                    )
                    admitted.append(pl)
                    det_index += 1

            if has_admitted:
                images_with_admitted += 1

        done = min(batch_start + batch, n_total)
        print(f"    {done:>6}/{n_total}  admitted so far: {len(admitted):,}",
              end="\r", flush=True)

    print()

    # ── Per-class audit stats ─────────────────────────────────────────────────
    def _mean(v: List[float]) -> Optional[float]:
        return sum(v) / len(v) if v else None

    per_class = {}
    for cid in range(n_classes):
        name = class_names[cid]
        per_class[name] = {
            "class_id":           cid,
            "det_count":          det_per_class.get(cid, 0),
            "admitted_count":     admitted_per_class.get(cid, 0),
            "det_pct":            det_per_class.get(cid, 0) / max(total_det, 1) * 100,
            "admitted_pct":       admitted_per_class.get(cid, 0) / max(len(admitted), 1) * 100,
            "mean_conf_all":      _mean(conf_per_class.get(cid, [])),
            "mean_conf_admitted": _mean(adm_conf_per_class.get(cid, [])),
        }

    top_class = max(per_class, key=lambda k: per_class[k]["admitted_count"],
                    default="?")

    audit_stats = {
        "round_id":                 round_id,
        "inference_conf":           inference_conf,
        "tau_conf":                 tau_conf,
        "n_unlabeled":              n_total,
        "images_with_any_det":      images_with_any_det,
        "images_with_admitted_det": images_with_admitted,
        "saturation_pct":           images_with_admitted / max(n_total, 1) * 100,
        "total_det_before_tau":     total_det,
        "admitted_count":           len(admitted),
        "top_class":                top_class,
        "top_class_pct":            per_class[top_class]["admitted_pct"] if top_class in per_class else 0,
        "per_class":                per_class,
    }

    # Print summary
    print(f"    admitted: {len(admitted):,} boxes from "
          f"{images_with_admitted:,}/{n_total:,} images "
          f"({audit_stats['saturation_pct']:.1f}% saturation)")
    print(f"    top class: {top_class} ({audit_stats['top_class_pct']:.1f}%)")
    for name, s in per_class.items():
        print(f"      {name:<14}  adm={s['admitted_count']:>6,}  "
              f"({s['admitted_pct']:.1f}%)  "
              f"mean_conf_adm={s['mean_conf_admitted'] or 0:.4f}")

    return admitted, audit_stats


# ── Main runner ───────────────────────────────────────────────────────────────

def run_baseline_b(split_file:       str,
                   init_checkpoint:  str,
                   output_dir:       str,
                   rounds:           int   = 3,
                   tau_conf:         float = 0.50,
                   inference_conf:   float = 0.05,
                   epochs:           int   = 100,
                   patience:         int   = 20,
                   optimizer:        str   = "AdamW",
                   lr0:              float = 0.001,
                   batch:            int   = 16,
                   imgsz:            int   = 640,
                   device:           str   = "0",
                   workers:          int   = 8,
                   infer_batch:      int   = 32) -> List[Dict]:
    """
    Run naive confidence-only iterative pseudo-labeling baseline.

    Returns list of per-round summary dicts.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Load split ────────────────────────────────────────────────────────────
    print(f"\nLoading split: {split_file}")
    split = load_split(split_file)
    class_names  = split["class_names"]
    num_classes  = len(class_names)
    unlabeled_ids = split["unlabeled_ids"]
    labeled_data  = filtered_split_to_labeled_data(split)
    image_dir     = Path(labeled_data["image_dir"])

    print(f"  schema=filtered  ratio={split['ratio']:.0%}  fold={split['fold']}")
    print(f"  labeled_train={split['n_labeled_train']:,}  "
          f"inner_val={split['n_labeled_inner_val']:,}  "
          f"unlabeled={len(unlabeled_ids):,}")
    print(f"  classes: {class_names}")
    print(f"  τ_conf={tau_conf}  inference_conf={inference_conf}")

    print()
    print("=" * 70)
    print("  Baseline B — Naive iterative pseudo-labeling")
    print("=" * 70)
    print(f"  init_checkpoint: {init_checkpoint}")
    print(f"  rounds: {rounds}  epochs/round: {epochs}  patience: {patience}")
    print()

    current_checkpoint = init_checkpoint
    all_summaries: List[Dict]  = []
    all_audit_stats: List[Dict] = []

    for round_id in range(1, rounds + 1):
        print()
        print(f"{'─'*70}")
        print(f"  Round {round_id}/{rounds}")
        print(f"{'─'*70}")
        round_dir = output_dir_path / f"round_{round_id:02d}"
        round_dir.mkdir(exist_ok=True)

        # ── Step 1: Generate pseudo-labels ────────────────────────────────────
        pseudo_labels, audit_stats = generate_pseudo_labels(
            checkpoint     = current_checkpoint,
            image_dir      = image_dir,
            unlabeled_ids  = unlabeled_ids,
            inference_conf = inference_conf,
            tau_conf       = tau_conf,
            imgsz          = imgsz,
            batch          = infer_batch,
            device         = device,
            round_id       = round_id,
            class_names    = class_names,
        )
        all_audit_stats.append(audit_stats)

        # Save round audit
        audit_path = round_dir / "pseudo_label_audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_stats, f, indent=2)

        # ── Step 2: Train ──────────────────────────────────────────────────────
        print(f"\n  Training round {round_id} from {Path(current_checkpoint).name} ...")
        print(f"    labeled_train: {split['n_labeled_train']:,} images  "
              f"+ {len(pseudo_labels):,} pseudo-labels")

        cfg = TrainingConfig(
            epochs      = epochs,
            patience    = patience,
            optimizer   = optimizer,
            lr0         = lr0,
            batch       = batch,
            imgsz       = imgsz,
            device      = device,
            workers     = workers,
            pretrained  = True,   # warm-start from previous .pt checkpoint
            base_model  = current_checkpoint,
            output_dir  = str(round_dir / "training"),
            num_classes = num_classes,
            class_names = class_names,
        )

        trainer    = YOLOTrainer(cfg)
        checkpoint = trainer.train(
            labeled_data  = labeled_data,
            pseudo_labels = pseudo_labels,
            round_id      = round_id,
        )
        print(f"  Checkpoint: {checkpoint}")

        # ── Step 3: Evaluate ───────────────────────────────────────────────────
        val_image_dir = split["image_dir_val"]
        val_label_dir = split["label_dir_val"]
        print(f"\n  Evaluating round {round_id} on filtered val2017 ...")

        result = evaluate_checkpoint(
            checkpoint    = checkpoint,
            val_image_dir = val_image_dir,
            val_label_dir = val_label_dir,
            num_classes   = num_classes,
            class_names   = class_names,
            imgsz         = imgsz,
            batch         = batch,
            device        = device,
            round_id      = round_id,
        )
        print(f"  mAP50:    {result.map50:.4f}")
        print(f"  mAP50-95: {result.map50_95:.4f}")
        if result.per_class:
            print(f"  Per-class AP50:")
            for name, ap in result.per_class.items():
                print(f"    {name}: {ap:.4f}")

        # ── Step 4: Save round summary ─────────────────────────────────────────
        round_summary = {
            "baseline":            "B_naive_iterative",
            "round_id":            round_id,
            "init_checkpoint":     current_checkpoint,
            "checkpoint":          checkpoint,
            "tau_conf":            tau_conf,
            "n_pseudo_labels":     len(pseudo_labels),
            "n_pseudo_images":     audit_stats["images_with_admitted_det"],
            "saturation_pct":      audit_stats["saturation_pct"],
            "top_class":           audit_stats["top_class"],
            "top_class_pct":       audit_stats["top_class_pct"],
            "map50":               result.map50,
            "map50_95":            result.map50_95,
            "per_class":           result.per_class,
            "created_at":          datetime.now(timezone.utc).isoformat(),
        }
        all_summaries.append(round_summary)

        # Save round result JSON
        round_result_path = round_dir / "summary.json"
        with open(round_result_path, "w") as f:
            json.dump(round_summary, f, indent=2)

        save_results(
            results     = [result],
            output_file = str(round_dir / "eval_result.json"),
            metadata    = round_summary,
        )

        # Advance checkpoint for next round
        current_checkpoint = checkpoint

    # ── Save full multi-round summary ─────────────────────────────────────────
    full_summary = {
        "baseline":          "B_naive_iterative",
        "split_file":        split_file,
        "init_checkpoint":   init_checkpoint,
        "tau_conf":          tau_conf,
        "inference_conf":    inference_conf,
        "rounds":            rounds,
        "epochs_per_round":  epochs,
        "patience":          patience,
        "optimizer":         optimizer,
        "lr0":               lr0,
        "class_names":       class_names,
        "rounds_summary":    all_summaries,
        "rounds_audit":      all_audit_stats,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    # ── Final print ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Baseline B complete")
    print("=" * 70)
    print(f"  {'Round':<8} {'mAP50':>8} {'mAP50-95':>10} "
          f"{'PL boxes':>10} {'sat%':>8}")
    print(f"  {'A (R0)':<8} {'0.4120':>8} {'0.2466':>10} {'—':>10} {'—':>8}")
    for s in all_summaries:
        print(f"  {s['round_id']:<8} {s['map50']:>8.4f} {s['map50_95']:>10.4f} "
              f"{s['n_pseudo_labels']:>10,} {s['saturation_pct']:>7.1f}%")
    print()
    print(f"  Full summary: {summary_path}")

    return all_summaries


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline B: naive confidence-only iterative pseudo-labeling"
    )
    parser.add_argument("--split_file",      required=True,
        help="Filtered split JSON (data/splits_vehicles/...)")
    parser.add_argument("--init_checkpoint", required=True,
        help="Round-0 checkpoint to warm-start from (use Baseline A best.pt)")
    parser.add_argument("--output_dir",      required=True,
        help="Directory to save per-round checkpoints and results")
    parser.add_argument("--rounds",     type=int,   default=3,
        help="Number of iterative rounds (default: 3)")
    parser.add_argument("--tau_conf",   type=float, default=0.50,
        help="Admission confidence threshold (default: 0.50)")
    parser.add_argument("--inference_conf", type=float, default=0.05,
        help="Inference conf for NMS (default: 0.05 — do not set to tau)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--optimizer",  default="AdamW",
        choices=["AdamW", "SGD", "Adam"])
    parser.add_argument("--lr0",        type=float, default=0.001)
    parser.add_argument("--batch",      type=int,   default=16)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--device",
        default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers",    type=int,   default=8)
    parser.add_argument("--infer_batch", type=int,  default=32,
        help="Batch size for inference pass (default: 32; "
             "larger = faster but more VRAM)")

    args = parser.parse_args()

    summaries = run_baseline_b(
        split_file      = args.split_file,
        init_checkpoint = args.init_checkpoint,
        output_dir      = args.output_dir,
        rounds          = args.rounds,
        tau_conf        = args.tau_conf,
        inference_conf  = args.inference_conf,
        epochs          = args.epochs,
        patience        = args.patience,
        optimizer       = args.optimizer,
        lr0             = args.lr0,
        batch           = args.batch,
        imgsz           = args.imgsz,
        device          = args.device,
        workers         = args.workers,
        infer_batch     = args.infer_batch,
    )