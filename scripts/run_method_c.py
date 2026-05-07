#!/usr/bin/env python3
"""
scripts/run_method_c.py
=======================
Method C: Stability-gated iterative auto-labeling.

Same training budget as Baseline B, but with:
  - Hungarian matching to track boxes across rounds
  - Stability score S(p,t) = α·C_cls_dist + β·C_round_loc + γ·C_jitter_loc
  - Dual admission gate: S >= τ_stab AND conf >= τ_conf
  - RawChurn/StableYield/ClassDrift stopping signals
  - Full bank entries with per-box component evidence

File-path-based inference (Path B) to avoid orchestrator RAM issue.
Uses all tested modules directly: canonical_infer, matcher, scorer,
admission, stopping, bank, trainer.

Round protocol (fair comparison with Baseline B):
  Round 0  =  Baseline A checkpoint + canonical inference on unlabeled → C_0 stored
  Round 1  =  infer → match vs C_0 → admit → train with PLs
  Round 2  =  infer → match vs C_1 → admit → train with PLs
  Round 3  =  infer → match vs C_2 → admit → train with PLs → evaluate

Usage:
  python scripts/run_method_c.py \\
      --split_json   data/splits_vehicles/coco_vehicles_fold0_10pct.json \\
      --init_checkpoint outputs/baseline_a/vehicles_fold0_10pct/training/round_0000/train/weights/best.pt \\
      --output_dir   outputs/method_c/vehicles_fold0_10pct \\
      --rounds       3 \\
      --epochs       100 \\
      --patience     20 \\
      --device       0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ── Project root on sys.path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from autolabel.bank.bank import PseudoLabelBank
from autolabel.bank.schemas import (
    BankEntry,
    CandidateIndex,
    CandidatePool,
    MatchResult,
    PseudoLabel,
    RoundMetadata,
    ScoringResult,
    StoppingSnapshot,
    StoppingState,
)
from autolabel.inference.canonical_infer import canonical_infer
from autolabel.inference.jitter import apply_jitter
from autolabel.matching.matcher import HungarianMatcher, MatchingConfig
from autolabel.scoring.scorer import ScoringConfig, StabilityScorer
from autolabel.admission.global_threshold import AdmissionConfig, GlobalThresholdPolicy
from autolabel.stopping.stopping import StoppingConfig, StoppingEvaluator
from autolabel.training.trainer import TrainingConfig, YOLOTrainer
from autolabel.utils.iou import iou_pair
from data.make_splits import load_split, filtered_split_to_labeled_data
from eval.evaluate import evaluate_checkpoint, save_results

logger = logging.getLogger("method_c")


# ═════════════════════════════════════════════════════════════════════════════
# Chunked canonical inference  (avoids loading all unlabeled images into RAM)
# ═════════════════════════════════════════════════════════════════════════════

def canonical_infer_from_paths(
    model: YOLO,
    image_paths: List[str],
    round_id: int,
    conf: float = 0.05,
    iou_nms: float = 0.45,
    imgsz: int = 640,
    batch_size: int = 8,
    chunk_size: int = 200,
) -> CandidatePool:
    """
    Canonical inference on images loaded from file paths in chunks.

    Loads `chunk_size` images at a time, runs canonical_infer, then frees
    the numpy arrays before loading the next chunk.  Peak RAM ≈ chunk_size
    images (~200 MB at chunk_size=200) instead of all unlabeled (~16 GB).

    Image IDs are derived from filename stems (e.g. "000000012345").
    Paths are sorted before processing to satisfy the batch composition
    contract in canonical_infer.
    """
    pool: CandidatePool = {}
    sorted_paths = sorted(image_paths)
    total = len(sorted_paths)

    for chunk_start in range(0, total, chunk_size):
        chunk_paths = sorted_paths[chunk_start : chunk_start + chunk_size]
        chunk_ids: List[str] = []
        chunk_imgs: List[np.ndarray] = []

        for p in chunk_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(
                    f"canonical_infer_from_paths: cv2.imread returned None "
                    f"for {p}. Check file exists and is a valid image."
                )
            chunk_ids.append(Path(p).stem)
            chunk_imgs.append(img)

        chunk_pool = canonical_infer(
            model=model,
            image_ids=chunk_ids,
            images=chunk_imgs,
            round_id=round_id,
            conf=conf,
            iou_nms=iou_nms,
            imgsz=imgsz,
            batch_size=batch_size,
        )
        pool.update(chunk_pool)
        del chunk_imgs  # free before next chunk

        done = min(chunk_start + chunk_size, total)
        logger.info("  canonical inference: %d / %d images", done, total)

    return pool


# ═════════════════════════════════════════════════════════════════════════════
# Jitter — per-image batched  (Fix 5: avoids redundant loads + inference)
# ═════════════════════════════════════════════════════════════════════════════

JITTER_TRANSFORMS = ["scale", "translate", "brightness"]


def compute_jitter_ious(
    model: YOLO,
    match_results: Dict[str, MatchResult],
    idx_t: CandidateIndex,
    idx_prev: CandidateIndex,
    id_to_path: Dict[str, str],
    tau_pre: float,
    min_iou_gate: float,
    conf: float = 0.05,
    iou_nms: float = 0.45,
    imgsz: int = 640,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute jitter IoUs for all qualifying matched pairs.

    Per-image batching: for each image that has ≥1 qualifying box, load the
    image once, apply each of the 3 transforms once, run YOLO 3 times total,
    then match all qualifying boxes from that image against each jittered
    output.  This is O(images × 3) inference calls, not O(boxes × 3).

    Frozen jitter detection selection rule (Patch 3), per box per transform:
      1. Among jittered-image detections, keep only same pred_class as pl_t
      2. Select detection with highest IoU vs remapped base box
      3. If none qualifies or best IoU < min_iou_gate: IoU_k = 0.0
    """
    jitter_ious: Dict[str, Tuple[float, float, float]] = {}

    # ── Group qualifying boxes by image ──────────────────────────────────────
    # qualifying_by_image[image_id] = [(box_id_t, pl_t, c_round_loc), ...]
    qualifying_by_image: Dict[str, List[Tuple[str, PseudoLabel, float]]] = {}

    for image_id, mr in match_results.items():
        for box_id_t, box_id_prev in mr.matched_pairs:
            pl_t = idx_t.get(box_id_t)
            pl_prev = idx_prev.get(box_id_prev)
            if pl_t is None or pl_prev is None:
                continue

            c_round_loc = iou_pair(pl_t.box, pl_prev.box)
            if c_round_loc < tau_pre:
                continue  # absent → scorer gives C_jitter_loc = 0.0

            if image_id not in qualifying_by_image:
                qualifying_by_image[image_id] = []
            qualifying_by_image[image_id].append((box_id_t, pl_t, c_round_loc))

    n_images = len(qualifying_by_image)
    n_boxes = sum(len(v) for v in qualifying_by_image.values())
    logger.info("  jitter: %d qualifying boxes across %d images", n_boxes, n_images)

    # ── Process per image ────────────────────────────────────────────────────
    for img_idx, (image_id, box_tuples) in enumerate(
            sorted(qualifying_by_image.items())):
        img_path = id_to_path.get(image_id)
        if img_path is None:
            logger.warning("jitter: no path for image %s", image_id)
            continue

        image = cv2.imread(img_path)
        if image is None:
            logger.warning("jitter: cv2.imread failed for %s", img_path)
            continue

        # Collect all base boxes for this image
        base_boxes = [pl_t.box for _, pl_t, _ in box_tuples]

        # Per-transform: jitter image once, infer once, match all boxes
        transform_ious: Dict[str, Dict[str, float]] = {
            t: {} for t in JITTER_TRANSFORMS
        }

        for transform in JITTER_TRANSFORMS:
            jittered_img, remapped_boxes = apply_jitter(
                image, base_boxes, transform)

            # Single inference call for this image + transform
            jitter_pool = canonical_infer(
                model=model,
                image_ids=[image_id],
                images=[jittered_img],
                round_id=box_tuples[0][1].round_id,
                conf=conf,
                iou_nms=iou_nms,
                imgsz=imgsz,
                batch_size=1,
            )
            jitter_dets = jitter_pool.get(image_id, [])

            # Match each qualifying box against jittered detections
            for box_idx, (box_id_t, pl_t, _) in enumerate(box_tuples):
                remapped_box = remapped_boxes[box_idx]

                # Frozen selection rule: same class, highest IoU
                same_class = [d for d in jitter_dets
                              if d.pred_class == pl_t.pred_class]
                if not same_class:
                    transform_ious[transform][box_id_t] = 0.0
                    continue

                best_iou = max(
                    iou_pair(d.box, remapped_box) for d in same_class)
                transform_ious[transform][box_id_t] = (
                    best_iou if best_iou >= min_iou_gate else 0.0
                )

        # Assemble triples — frozen convention: triple only written if
        # ALL 3 transforms clear min_iou_gate.  Otherwise absent from
        # jitter_ious → scorer assigns C_jitter_loc=0.0, jitter_executed=False.
        for box_id_t, _, _ in box_tuples:
            vals = (
                transform_ious["scale"].get(box_id_t, 0.0),
                transform_ious["translate"].get(box_id_t, 0.0),
                transform_ious["brightness"].get(box_id_t, 0.0),
            )
            if all(v >= min_iou_gate for v in vals):
                jitter_ious[box_id_t] = vals
            # else: absent → scorer gives C_jitter_loc=0.0, jitter_executed=False

        if (img_idx + 1) % 500 == 0 or (img_idx + 1) == n_images:
            logger.info("  jitter progress: %d / %d images", img_idx + 1,
                        n_images)

    return jitter_ious


# ═════════════════════════════════════════════════════════════════════════════
# Bank write helper  (Step 7)
# ═════════════════════════════════════════════════════════════════════════════

def write_round_to_bank(
    bank: PseudoLabelBank,
    C_t: CandidatePool,
    C_prev: CandidatePool,
    match_results: Dict[str, MatchResult],
    scoring_result: ScoringResult,
    admitted_ids: set,
    round_t: int,
) -> None:
    """Write per-image BankEntries for this round (append-only)."""
    all_image_ids = set(C_t.keys()) | set(C_prev.keys())

    for image_id in sorted(all_image_ids):
        pls = C_t.get(image_id, [])
        mr = match_results.get(image_id)

        # Build prev-box lookup from matched_pairs
        prev_map: Dict[str, Optional[str]] = {}
        if mr is not None:
            for bt, bp in mr.matched_pairs:
                prev_map[bt] = bp

        bank.append(BankEntry(
            image_id=image_id,
            round_id=round_t,
            pseudo_labels=pls,
            stability_scores={
                pl.box_id: scoring_result.scores.get(pl.box_id)
                for pl in pls
            },
            admitted={
                pl.box_id: pl.box_id in admitted_ids
                for pl in pls
            },
            matched_prev_box_id={
                pl.box_id: prev_map.get(pl.box_id)
                for pl in pls
            },
            is_ambiguous={
                pl.box_id: (pl.box_id in mr.ambiguous if mr else False)
                for pl in pls
            },
            c_cls_dist={
                pl.box_id: scoring_result.components[pl.box_id].c_cls_dist
                if pl.box_id in scoring_result.components else None
                for pl in pls
            },
            c_round_loc={
                pl.box_id: scoring_result.components[pl.box_id].c_round_loc
                if pl.box_id in scoring_result.components else None
                for pl in pls
            },
            c_jitter_loc={
                pl.box_id: scoring_result.components[pl.box_id].c_jitter_loc
                if pl.box_id in scoring_result.components else None
                for pl in pls
            },
            jitter_executed={
                pl.box_id: scoring_result.components[pl.box_id].jitter_executed
                if pl.box_id in scoring_result.components else False
                for pl in pls
            },
        ))


# ═════════════════════════════════════════════════════════════════════════════
# Round-0 bank bootstrap  (Fix 1: store C_0 so Round 1 can match against it)
# ═════════════════════════════════════════════════════════════════════════════

def write_bootstrap_round_to_bank(
    bank: PseudoLabelBank,
    C_0: CandidatePool,
    round_id: int = 0,
) -> None:
    """
    Write Round-0 candidate pool to bank WITHOUT scoring/matching.

    All fields that require cross-round comparison (stability_scores,
    matched_prev_box_id, component evidence) are set to None/False.
    This is correct: Round 0 has no prior round to compare against.
    The purpose is solely to populate C_prev for Round 1.
    """
    for image_id in sorted(C_0.keys()):
        pls = C_0[image_id]
        bank.append(BankEntry(
            image_id=image_id,
            round_id=round_id,
            pseudo_labels=pls,
            stability_scores={pl.box_id: None for pl in pls},
            admitted={pl.box_id: False for pl in pls},
            matched_prev_box_id={pl.box_id: None for pl in pls},
            is_ambiguous={pl.box_id: False for pl in pls},
            c_cls_dist={pl.box_id: None for pl in pls},
            c_round_loc={pl.box_id: None for pl in pls},
            c_jitter_loc={pl.box_id: None for pl in pls},
            jitter_executed={pl.box_id: False for pl in pls},
        ))


# ═════════════════════════════════════════════════════════════════════════════
# Per-class pseudo-label stats  (same format as baseline_b for comparison)
# ═════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["bicycle", "car", "motorcycle", "bus", "truck"]


def log_pl_stats(
    A_t: List[PseudoLabel], C_t: CandidatePool, round_t: int
) -> None:
    """Log per-class pseudo-label statistics for this round."""
    n_candidates = sum(len(v) for v in C_t.values())
    n_admitted = len(A_t)
    images_with_pl = len({pl.image_id for pl in A_t})

    logger.info("  Round %d PL stats: %d admitted / %d candidates (%.1f%%)",
                round_t, n_admitted, n_candidates,
                100.0 * n_admitted / n_candidates if n_candidates > 0 else 0.0)
    logger.info("  Images with PLs: %d", images_with_pl)

    # Per-class breakdown
    class_counts: Dict[int, int] = {}
    for pl in A_t:
        class_counts[pl.pred_class] = class_counts.get(pl.pred_class, 0) + 1
    for cls_id in range(len(CLASS_NAMES)):
        count = class_counts.get(cls_id, 0)
        pct = 100.0 * count / n_admitted if n_admitted > 0 else 0.0
        logger.info("    %s: %d (%.1f%%)", CLASS_NAMES[cls_id], count, pct)


# ═════════════════════════════════════════════════════════════════════════════
# Main round loop
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Method C: Stability-gated iterative auto-labeling")
    parser.add_argument("--split_json", required=True,
                        help="Path to filtered split JSON")
    parser.add_argument("--init_checkpoint", required=True,
                        help="Baseline A best.pt (Round 0 checkpoint)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for this experiment")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of auto-labeling rounds (default: 3)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--chunk_size", type=int, default=200,
                        help="Images loaded per chunk (RAM control)")
    parser.add_argument("--max_unlabeled", type=int, default=None,
                        help="Limit unlabeled images for smoke testing")

    # ── Stability hyperparameters (architecture defaults) ─────────────────
    parser.add_argument("--alpha", type=float, default=0.33)
    parser.add_argument("--beta", type=float, default=0.33)
    parser.add_argument("--gamma", type=float, default=0.34)
    parser.add_argument("--tau_stab", type=float, default=0.6,
                        help="Minimum stability score for admission")
    parser.add_argument("--tau_conf", type=float, default=0.5,
                        help="Minimum confidence for admission "
                             "(0.50 matches Baseline B for fair comparison)")
    parser.add_argument("--min_iou_gate", type=float, default=0.3,
                        help="Matching IoU gate + jitter prefilter + "
                             "jitter selection")
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="JS weight in matching cost")
    parser.add_argument("--epsilon_match", type=float, default=0.05,
                        help="Ambiguity rejection gap threshold")
    parser.add_argument("--warmup_rounds", type=int, default=2,
                        help="Disable stopping for first N rounds")
    parser.add_argument("--K_consecutive", type=int, default=2,
                        help="Consecutive rounds to trigger stop")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete output_dir before starting (prevents "
                             "stale bank corruption on rerun)")

    args = parser.parse_args()

    # ── Stale output guard ──────────────────────────────────────────────────
    import shutil
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise RuntimeError(
                f"Output dir already exists: {args.output_dir}. "
                f"Use --overwrite or choose a new --output_dir."
            )

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "method_c.log"), mode="w"),
        ],
    )
    logger.info("Method C — Stability-gated iterative auto-labeling")
    logger.info("Args: %s", vars(args))

    # ── Load split ───────────────────────────────────────────────────────────
    split = load_split(args.split_json)
    labeled_data = filtered_split_to_labeled_data(split)
    logger.info("Labeled images: %d, Inner-val: %d, Unlabeled: %d",
                len(split["labeled_train_ids"]),
                len(split["labeled_inner_val_ids"]),
                len(split["unlabeled_ids"]))

    # ── Build unlabeled image paths + hard validation (Fix 4) ────────────────
    image_dir = split["image_dir_train"]
    unlabeled_paths: List[str] = []
    for img_id in split["unlabeled_ids"]:
        # COCO convention: integer ID → 000000000064.jpg (12-digit zero-padded)
        fname = f"{int(img_id):012d}.jpg"
        p = os.path.join(image_dir, fname)
        unlabeled_paths.append(p)

    if args.max_unlabeled is not None:
        unlabeled_paths = unlabeled_paths[:args.max_unlabeled]
        logger.info("Smoke mode: limited to %d unlabeled images",
                    len(unlabeled_paths))

    missing = [p for p in unlabeled_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} unlabeled images. "
            f"First 5: {missing[:5]}"
        )
    logger.info("Unlabeled image paths validated: %d (all exist)",
                len(unlabeled_paths))

    id_to_path: Dict[str, str] = {Path(p).stem: p for p in unlabeled_paths}

    # ── Instantiate modules ──────────────────────────────────────────────────
    matcher = HungarianMatcher(MatchingConfig(
        lambda_cls=args.lambda_cls,
        min_iou_gate=args.min_iou_gate,
        epsilon_match=args.epsilon_match,
    ))

    scorer = StabilityScorer(ScoringConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        tau_pre=args.min_iou_gate,  # coupling constraint: must == min_iou_gate
    ))

    policy = GlobalThresholdPolicy(AdmissionConfig(
        tau_stab=args.tau_stab,
        tau_conf=args.tau_conf,
    ))

    stopper = StoppingEvaluator(StoppingConfig(
        epsilon_churn=0.02,
        epsilon_yield=0.03,
        epsilon_drift=0.01,
        tau_yield=0.05,
        tau_drift=0.10,
        tau_geom_change=0.50,
        K_consecutive=args.K_consecutive,
    ))

    bank_path = os.path.join(args.output_dir, "bank")
    bank = PseudoLabelBank.load_or_create(bank_path)

    # Training config is created fresh each round (like Baseline B):
    # base_model = current checkpoint, pretrained = True, num_classes/class_names set.
    # This prevents Ultralytics from rebuilding the Detect head to a different nc.

    stop_state = StoppingState()

    # ═════════════════════════════════════════════════════════════════════════
    # Round 0: Bootstrap — canonical inference with Baseline A checkpoint
    # (Fix 1: populate C_0 in bank so Round 1 has something to match against)
    # ═════════════════════════════════════════════════════════════════════════
    checkpoint_t = args.init_checkpoint
    logger.info("=" * 70)
    logger.info("ROUND 0: Bootstrap — canonical inference with Baseline A")
    logger.info("=" * 70)
    logger.info("Checkpoint: %s", checkpoint_t)

    model_0 = YOLO(checkpoint_t)
    C_0 = canonical_infer_from_paths(
        model=model_0,
        image_paths=unlabeled_paths,
        round_id=0,
        conf=0.05,
        iou_nms=0.45,
        imgsz=640,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    n_c0 = sum(len(v) for v in C_0.values())
    logger.info("  C_0: %d candidates across %d images", n_c0, len(C_0))

    write_bootstrap_round_to_bank(bank, C_0, round_id=0)
    bank.write_metadata(RoundMetadata(
        round_id=0,
        n_candidates=n_c0,
        n_admitted=0,
        raw_churn=0.0,
        stable_yield=0.0,
        class_drift=0.0,
        stop_condition_met=False,
        model_checkpoint=checkpoint_t,
    ))
    del model_0
    torch.cuda.empty_cache()
    logger.info("  Round 0 bootstrap complete: C_0 stored in bank.")

    # ═════════════════════════════════════════════════════════════════════════
    # Round loop  (1 .. max_rounds)
    # ═════════════════════════════════════════════════════════════════════════
    for round_t in range(1, args.rounds + 1):
        round_start = time.time()
        logger.info("=" * 70)
        logger.info("ROUND %d / %d", round_t, args.rounds)
        logger.info("=" * 70)

        # ── STEP 1: Canonical inference ──────────────────────────────────────
        logger.info("Step 1: Canonical inference on %d unlabeled images...",
                    len(unlabeled_paths))
        model_t = YOLO(checkpoint_t)
        C_t = canonical_infer_from_paths(
            model=model_t,
            image_paths=unlabeled_paths,
            round_id=round_t,
            conf=0.05,
            iou_nms=0.45,
            imgsz=640,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        idx_t = CandidateIndex.build(C_t)
        n_candidates = sum(len(v) for v in C_t.values())
        logger.info("  C_t: %d candidates across %d images",
                    n_candidates, len(C_t))

        # ── STEP 2: Retrieve prior-round candidates ─────────────────────────
        C_prev = bank.get_candidate_pool(round_id=round_t - 1)
        idx_prev = CandidateIndex.build(C_prev)
        n_prev = sum(len(v) for v in C_prev.values())
        logger.info("Step 2: C_prev (round %d): %d candidates across %d "
                    "images", round_t - 1, n_prev, len(C_prev))

        # ── STEP 3: Hungarian matching (per image) ──────────────────────────
        logger.info("Step 3: Hungarian matching...")
        match_results: Dict[str, MatchResult] = {}
        all_image_ids = set(C_t.keys()) | set(C_prev.keys())
        for image_id in all_image_ids:
            match_results[image_id] = matcher.match(
                current=C_t.get(image_id, []),
                previous=C_prev.get(image_id, []),
            )

        n_matched = sum(
            len(mr.matched_pairs) for mr in match_results.values())
        n_ambiguous = sum(
            len(mr.ambiguous) for mr in match_results.values())
        n_unmatched = sum(
            len(mr.unmatched_current) for mr in match_results.values())
        logger.info("  matched: %d, ambiguous: %d, unmatched_current: %d",
                    n_matched, n_ambiguous, n_unmatched)

        # ── STEP 4: Jitter inference ─────────────────────────────────────────
        logger.info("Step 4: Jitter inference (τ_pre=%.2f)...",
                    args.min_iou_gate)
        jitter_ious = compute_jitter_ious(
            model=model_t,
            match_results=match_results,
            idx_t=idx_t,
            idx_prev=idx_prev,
            id_to_path=id_to_path,
            tau_pre=args.min_iou_gate,
            min_iou_gate=args.min_iou_gate,
            conf=0.05,
            iou_nms=0.45,
            imgsz=640,
        )
        logger.info("  jitter IoUs computed for %d boxes", len(jitter_ious))

        # ── STEP 5: Stability scoring ────────────────────────────────────────
        logger.info("Step 5: Stability scoring (α=%.2f, β=%.2f, γ=%.2f)...",
                    args.alpha, args.beta, args.gamma)
        scoring_result: ScoringResult = scorer.score(
            C_t=C_t,
            C_prev=C_prev,
            match_results=match_results,
            jitter_ious=jitter_ious,
        )

        scored_vals = [v for v in scoring_result.scores.values()
                       if v is not None]
        if scored_vals:
            logger.info("  scored: %d boxes, mean S=%.3f, median S=%.3f",
                        len(scored_vals),
                        sum(scored_vals) / len(scored_vals),
                        sorted(scored_vals)[len(scored_vals) // 2])
        else:
            logger.info("  scored: 0 boxes (no matched pairs)")

        # ── STEP 6: Admission ────────────────────────────────────────────────
        logger.info("Step 6: Admission (τ_stab=%.2f, τ_conf=%.2f)...",
                    args.tau_stab, args.tau_conf)
        A_t = policy.admit(
            candidates=C_t,
            scores=scoring_result.scores,
            round_id=round_t,
        )
        admitted_ids = {pl.box_id for pl in A_t}
        log_pl_stats(A_t, C_t, round_t)

        # ── STEP 7: Write to bank ────────────────────────────────────────────
        logger.info("Step 7: Writing to bank...")
        write_round_to_bank(
            bank=bank,
            C_t=C_t,
            C_prev=C_prev,
            match_results=match_results,
            scoring_result=scoring_result,
            admitted_ids=admitted_ids,
            round_t=round_t,
        )

        # ── STEP 8: Stopping evaluation ──────────────────────────────────────
        logger.info("Step 8: Stopping signals...")
        if round_t > args.warmup_rounds:
            stop_state = stopper.evaluate(
                C_t=C_t,
                C_prev=C_prev,
                A_t=A_t,
                match_results=match_results,
                stop_state=stop_state,
                round_t=round_t,
            )
        else:
            snapshot = stopper.compute_signals(
                C_t, C_prev, A_t, match_results, round_t)
            stop_state.signal_history.append(snapshot)

        last = stop_state.signal_history[-1]
        logger.info("  RawChurn=%.4f  StableYield=%.4f  ClassDrift=%.4f",
                    last.raw_churn, last.stable_yield, last.class_drift)
        logger.info("  consecutive_satisfied=%d  stopped=%s",
                    stop_state.consecutive_satisfied, stop_state.stopped)

        # ── STEP 9: Log round metadata ───────────────────────────────────────
        bank.write_metadata(RoundMetadata(
            round_id=round_t,
            n_candidates=n_candidates,
            n_admitted=len(A_t),
            raw_churn=last.raw_churn,
            stable_yield=last.stable_yield,
            class_drift=last.class_drift,
            stop_condition_met=stop_state.consecutive_satisfied > 0,
            model_checkpoint=checkpoint_t,
        ))

        # ── STEP 10: Check convergence stop ──────────────────────────────────
        # NOTE: do NOT break here for max_rounds — always train first (Fix 2)
        if stop_state.stopped:
            logger.info("CONVERGENCE STOP at round %d: %s",
                        round_t, stop_state.stop_reason)
            # Still train on this round's admissions before exiting
            logger.info("Step 11: Final training (convergence round)...")
            del model_t
            torch.cuda.empty_cache()
            round_train_dir = os.path.join(
                args.output_dir, "training", f"round_{round_t:04d}")
            cfg = TrainingConfig(
                epochs=args.epochs, patience=args.patience,
                optimizer="AdamW", lr0=0.001, batch=16, imgsz=640,
                device=args.device, workers=8,
                pretrained=True, base_model=checkpoint_t,
                output_dir=round_train_dir,
                num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES,
            )
            trainer = YOLOTrainer(cfg)
            checkpoint_t = trainer.train(
                labeled_data=labeled_data,
                pseudo_labels=A_t,
                round_id=round_t,
            )
            logger.info("  checkpoint: %s", checkpoint_t)
            break

        # ── STEP 11: Retrain ─────────────────────────────────────────────────
        logger.info("Step 11: Training round %d (%d PLs, warm-start)...",
                    round_t, len(A_t))

        del model_t
        torch.cuda.empty_cache()

        round_train_dir = os.path.join(
            args.output_dir, "training", f"round_{round_t:04d}")
        cfg = TrainingConfig(
            epochs=args.epochs, patience=args.patience,
            optimizer="AdamW", lr0=0.001, batch=16, imgsz=640,
            device=args.device, workers=8,
            pretrained=True, base_model=checkpoint_t,
            output_dir=round_train_dir,
            num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES,
        )
        trainer = YOLOTrainer(cfg)
        checkpoint_t = trainer.train(
            labeled_data=labeled_data,
            pseudo_labels=A_t,
            round_id=round_t,
        )
        logger.info("  checkpoint: %s", checkpoint_t)

        # ── Evaluate this round (Fix 3: fail loudly) ────────────────────────
        logger.info("Evaluating round %d checkpoint...", round_t)
        results = evaluate_checkpoint(
            checkpoint=checkpoint_t,
            val_image_dir=split["image_dir_val"],
            val_label_dir=split["label_dir_val"],
            num_classes=len(CLASS_NAMES),
            class_names=CLASS_NAMES,
            device=args.device,
            round_id=round_t,
        )
        save_results([results], os.path.join(
            args.output_dir, f"eval_round_{round_t:04d}.json"))
        logger.info("  mAP50=%.4f  mAP50-95=%.4f",
                    results.map50,
                    results.map50_95)

        round_elapsed = time.time() - round_start
        logger.info("Round %d complete in %.1f min",
                    round_t, round_elapsed / 60)

    # Mark max_rounds if loop ended without convergence stop
    if not stop_state.stopped:
        stop_state.stopped = True
        stop_state.stop_reason = "max_rounds"

    # ═════════════════════════════════════════════════════════════════════════
    # Final evaluation
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)
    logger.info("Final checkpoint: %s", checkpoint_t)
    logger.info("Stop reason: %s", stop_state.stop_reason)

    final_results = evaluate_checkpoint(
        checkpoint=checkpoint_t,
        val_image_dir=split["image_dir_val"],
        val_label_dir=split["label_dir_val"],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        device=args.device,
    )
    save_results([final_results], os.path.join(
        args.output_dir, "eval_final.json"))
    logger.info("Final mAP50=%.4f  mAP50-95=%.4f",
                final_results.map50,
                final_results.map50_95)

    # ── Signal history summary ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SIGNAL HISTORY")
    logger.info("%-6s  %-10s  %-12s  %-10s",
                "Round", "RawChurn", "StableYield", "ClassDrift")
    for snap in stop_state.signal_history:
        logger.info("%-6d  %-10.4f  %-12.4f  %-10.4f",
                    snap.round_id, snap.raw_churn,
                    snap.stable_yield, snap.class_drift)
    logger.info("=" * 70)
    logger.info("Done.")


if __name__ == "__main__":
    main()