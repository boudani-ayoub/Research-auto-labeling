"""
data/make_splits.py
===================
Creates labeled/unlabeled splits for the COCO low-label benchmark protocol.

Standard protocol (from SSOD literature — Humble Teacher, Dense Teacher):
  - Source: MS COCO train2017 (118,287 images)
  - Label ratios: 1%, 5%, 10% of train2017
  - Folds: 5 random folds per ratio (different random seeds)
  - Unlabeled: remainder of train2017 (not val2017)
  - Evaluation: val2017 (5,000 images) — fixed, not a split

Output files (one per fold per ratio):
  data/splits/coco_fold{0..4}_1pct.json
  data/splits/coco_fold{0..4}_5pct.json
  data/splits/coco_fold{0..4}_10pct.json

Each JSON file contains:
  {
    "ratio":                  0.01,
    "fold":                   0,
    "seed":                   42,
    "inner_val_seed":         542,           -- seed for the 90/10 sub-split
    "val_fraction":           0.10,          -- inner-val fraction of labeled set
    "n_labeled":              1182,          -- total labeled (= train + inner_val)
    "n_labeled_train":        1064,          -- 90% — used for YOLO training input
    "n_labeled_inner_val":    118,           -- 10% — used by YOLO for early stopping
    "n_unlabeled":            117105,
    "n_val":                  5000,
    "labeled_ids":            [int, ...],    -- union (kept for legacy / sanity)
    "labeled_train_ids":      [int, ...],    -- 90% — TRAINING input
    "labeled_inner_val_ids":  [int, ...],    -- 10% — INNER VAL (frozen, no pseudo-labels)
    "unlabeled_ids":          [int, ...],    -- COCO image_ids (unchanged)
    "val_ids":                [int, ...],    -- COCO image_ids from val2017 (final eval)
    "coco_train_dir":         str,
    "coco_val_dir":           str,
    "annotation_file":        str,
    "val_annotation_file":    str,
    "created_at":             str,
  }

Inner-val rationale (Option A protocol — frozen):
  YOLO needs a held-out set inside one training run for patience-based
  early stopping. We must NOT use COCO val2017 for that — val2017 is
  the final paper evaluation set and using it inside training is leakage.
  Instead we hold out 10% of the labeled set as an *inner* val.

  This inner val is used:
    * Inside Baseline A / Baseline B / Our method, for YOLO's per-run
      early stopping (`patience`).

  This inner val is NEVER used:
    * To pick admission thresholds (tau_stab, tau_conf).
    * To decide round count, stopping signals, or stopping thresholds.
    * To gate keep-or-revert decisions on round checkpoints.
  Our method's "validation-free" claim is about the OUTER LOOP only —
  see paper protocol section.

  Both Baseline A and Our method must train on the SAME labeled_train_ids
  and use the SAME labeled_inner_val_ids. This makes the asymmetric claim
  ("Baseline A uses validation, Our outer loop does not") clean.

Usage:
  python data/make_splits.py \\
      --coco_dir /path/to/research/coco \\
      --output_dir data/splits \\
      --ratios 0.01 0.05 0.10 \\
      --n_folds 5 \\
      --base_seed 42

  # For a single fold (testing):
  python data/make_splits.py \\
      --coco_dir /path/to/research/coco \\
      --output_dir data/splits \\
      --ratios 0.10 \\
      --n_folds 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


# ── Fold seeds ────────────────────────────────────────────────────────────────
# Fixed seeds per fold — reproducible across machines and runs.
# Fold seeds are derived from base_seed + fold_index so they are independent.

def fold_seed(base_seed: int, fold: int) -> int:
    return base_seed + fold * 1000


def inner_val_seed(base_seed: int, fold: int) -> int:
    """
    Seed for the inner-val 90/10 sub-split within the labeled set.

    Derived from fold_seed but offset by 500 so the inner-val sampling RNG
    is independent of the labeled/unlabeled sampling RNG. This guarantees
    that changing the inner-val fraction (e.g. 0.10 → 0.15) does not
    perturb the labeled/unlabeled split — they share base_seed but use
    distinct RNG instances.

    Convention:
        fold 0: base_seed=42  → fold_seed=42    → inner_val_seed=542
        fold 1: base_seed=42  → fold_seed=1042  → inner_val_seed=1542
        fold 2: base_seed=42  → fold_seed=2042  → inner_val_seed=2542
    """
    return fold_seed(base_seed, fold) + 500


# ── Inner-val sub-split (90/10) ───────────────────────────────────────────────

def split_labeled_into_train_val(labeled_ids:  List[int],
                                  base_seed:    int,
                                  fold:         int,
                                  val_fraction: float = 0.10
                                  ) -> Tuple[List[int], List[int]]:
    """
    Split the labeled set into training and inner-validation portions.

    The inner-val set is consumed by YOLO inside ONE training run (per round)
    for patience-based early stopping. It is NEVER consumed by the outer loop
    (admission, stopping, threshold tuning).

    Args:
        labeled_ids:   list of labeled image IDs (any order — sorted internally
                       before shuffling for determinism)
        base_seed:     base random seed
        fold:          fold index
        val_fraction:  fraction of labeled set held out as inner val (default 0.10)

    Returns:
        (labeled_train_ids, labeled_inner_val_ids) — each sorted ascending.

    Notes:
        At 1% labels with val_fraction=0.10, this yields ~1065 train / ~118
        inner val. 118 images is a tight inner-val signal for COCO-80 — this
        is accepted as an MVP limitation. v2 may use stratified sampling.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1), got {val_fraction}"
        )
    if len(labeled_ids) < 10:
        raise ValueError(
            f"labeled_ids has only {len(labeled_ids)} items; refusing to split "
            f"into train/inner-val (need at least 10)."
        )

    n_total = len(labeled_ids)
    n_val   = max(1, round(n_total * val_fraction))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError(
            f"Computed n_train={n_train} for n_total={n_total}, "
            f"val_fraction={val_fraction}. Increase labeled set or decrease "
            f"val_fraction."
        )

    # Canonical order before shuffle — independent of input list ordering.
    canonical = sorted(labeled_ids)
    shuffled  = canonical.copy()
    rng       = random.Random(inner_val_seed(base_seed, fold))
    rng.shuffle(shuffled)

    inner_val_ids = sorted(shuffled[:n_val])
    train_ids     = sorted(shuffled[n_val:])

    # Defensive invariant: union must equal input set, partition must be disjoint.
    assert set(train_ids).isdisjoint(set(inner_val_ids)), \
        "internal error: train and inner-val partitions overlap"
    assert set(train_ids) | set(inner_val_ids) == set(labeled_ids), \
        "internal error: partition does not cover input"

    return train_ids, inner_val_ids


# ── Split creation ────────────────────────────────────────────────────────────

def load_image_ids(annotation_file: str) -> List[int]:
    """
    Load all image_ids from a COCO annotation JSON file.
    Returns sorted list for deterministic ordering before sampling.
    """
    with open(annotation_file, "r") as f:
        data = json.load(f)
    ids = sorted([img["id"] for img in data["images"]])
    return ids


def make_split(all_train_ids: List[int],
               val_ids:       List[int],
               ratio:         float,
               fold:          int,
               base_seed:     int,
               coco_train_dir: str,
               coco_val_dir:   str,
               annotation_file: str,
               val_annotation_file: str,
               val_fraction:  float = 0.10) -> Dict:
    """
    Create one labeled/unlabeled split, with an additional 90/10 sub-split
    of the labeled set into (labeled_train_ids, labeled_inner_val_ids).

    Args:
        all_train_ids:        sorted list of all train2017 image_ids
        val_ids:              sorted list of all val2017 image_ids
        ratio:                fraction of train2017 to use as labeled (e.g. 0.01)
        fold:                 fold index 0..4
        base_seed:            base random seed
        coco_train_dir:       path to train2017 images directory
        coco_val_dir:         path to val2017 images directory
        annotation_file:      path to instances_train2017.json
        val_annotation_file:  path to instances_val2017.json
        val_fraction:         fraction of labeled set held out as inner val
                              for YOLO per-run early stopping (default 0.10).

    Returns:
        split dict ready for JSON serialization.
    """
    seed = fold_seed(base_seed, fold)
    rng  = random.Random(seed)

    n_total   = len(all_train_ids)
    n_labeled = max(1, round(n_total * ratio))

    # Shuffle a copy — do not modify the original sorted list
    shuffled = all_train_ids.copy()
    rng.shuffle(shuffled)

    labeled_ids   = sorted(shuffled[:n_labeled])
    unlabeled_ids = sorted(shuffled[n_labeled:])

    # ── Inner-val sub-split (90/10 by default) ───────────────────────────
    # Independent RNG (seeded by inner_val_seed) so val_fraction can be
    # changed without perturbing the labeled/unlabeled boundary above.
    labeled_train_ids, labeled_inner_val_ids = split_labeled_into_train_val(
        labeled_ids  = labeled_ids,
        base_seed    = base_seed,
        fold         = fold,
        val_fraction = val_fraction,
    )

    return {
        "ratio":                  ratio,
        "fold":                   fold,
        "seed":                   seed,
        "inner_val_seed":         inner_val_seed(base_seed, fold),
        "val_fraction":           val_fraction,
        "n_labeled":              len(labeled_ids),
        "n_labeled_train":        len(labeled_train_ids),
        "n_labeled_inner_val":    len(labeled_inner_val_ids),
        "n_unlabeled":            len(unlabeled_ids),
        "n_val":                  len(val_ids),
        "labeled_ids":            labeled_ids,
        "labeled_train_ids":      labeled_train_ids,
        "labeled_inner_val_ids":  labeled_inner_val_ids,
        "unlabeled_ids":          unlabeled_ids,
        "val_ids":                val_ids,
        "coco_train_dir":         str(coco_train_dir),
        "coco_val_dir":           str(coco_val_dir),
        "annotation_file":        str(annotation_file),
        "val_annotation_file":    str(val_annotation_file),
        "created_at":             datetime.now(timezone.utc).isoformat(),
    }


def ratio_to_str(ratio: float) -> str:
    """Convert ratio to filename suffix: 0.01 → '1pct', 0.05 → '5pct'."""
    pct = ratio * 100
    if pct == int(pct):
        return f"{int(pct)}pct"
    return f"{pct:.1f}pct".replace(".", "p")


def make_all_splits(coco_dir:    str,
                    output_dir:  str,
                    ratios:      List[float],
                    n_folds:     int,
                    base_seed:   int,
                    val_fraction: float = 0.10) -> None:
    """
    Create all splits for all ratios and folds. Write JSON files to output_dir.
    """
    coco_dir   = Path(coco_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir        = coco_dir / "images" / "train2017"
    val_dir          = coco_dir / "images" / "val2017"
    annotation_file  = coco_dir / "annotations" / "instances_train2017.json"
    val_ann_file     = coco_dir / "annotations" / "instances_val2017.json"

    # Validate paths
    for p, name in [(train_dir,       "train2017 images dir"),
                    (val_dir,         "val2017 images dir"),
                    (annotation_file, "instances_train2017.json"),
                    (val_ann_file,    "instances_val2017.json")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required COCO path not found: {p}\n"
                f"Expected: {name}\n"
                f"Verify --coco_dir points to the root of your COCO download."
            )

    print(f"Loading train image IDs from {annotation_file} ...")
    all_train_ids = load_image_ids(str(annotation_file))
    print(f"  train2017: {len(all_train_ids):,} images")

    print(f"Loading val image IDs from {val_ann_file} ...")
    val_ids = load_image_ids(str(val_ann_file))
    print(f"  val2017:   {len(val_ids):,} images")
    print(f"Inner-val fraction (per-run early stopping): {val_fraction:.0%}")
    print()

    created = []
    for ratio in ratios:
        ratio_str = ratio_to_str(ratio)
        n_labeled_expected = max(1, round(len(all_train_ids) * ratio))
        print(f"Ratio {ratio:.0%} → ~{n_labeled_expected:,} labeled images")

        for fold in range(n_folds):
            split = make_split(
                all_train_ids       = all_train_ids,
                val_ids             = val_ids,
                ratio               = ratio,
                fold                = fold,
                base_seed           = base_seed,
                coco_train_dir      = str(train_dir),
                coco_val_dir        = str(val_dir),
                annotation_file     = str(annotation_file),
                val_annotation_file = str(val_ann_file),
                val_fraction        = val_fraction,
            )

            out_path = output_dir / f"coco_fold{fold}_{ratio_str}.json"
            with open(out_path, "w") as f:
                json.dump(split, f, indent=2)

            print(f"  fold {fold}: "
                  f"{split['n_labeled_train']:,} train + "
                  f"{split['n_labeled_inner_val']:,} inner_val, "
                  f"{split['n_unlabeled']:,} unlabeled → {out_path.name}")
            created.append(out_path)

        print()

    print(f"Created {len(created)} split files in {output_dir}/")
    return created


# ── Validation helpers (used by orchestrator and trainer) ────────────────────

def load_split(split_file: str) -> Dict:
    """
    Load a split JSON file. Validates required fields are present.

    Splits without `labeled_train_ids` / `labeled_inner_val_ids` are legacy
    (pre-Option A protocol) and must be regenerated. The error message
    points to the regeneration command.
    """
    required = ["ratio", "fold", "labeled_ids", "unlabeled_ids",
                "labeled_train_ids", "labeled_inner_val_ids",
                "coco_train_dir", "coco_val_dir", "annotation_file"]
    with open(split_file, "r") as f:
        split = json.load(f)
    for field in required:
        if field not in split:
            raise ValueError(
                f"Split file {split_file!r} is missing required field: "
                f"{field!r}.\n"
                f"If this is a legacy split (pre-Option A inner-val protocol), "
                f"regenerate with:\n"
                f"  python data/make_splits.py --coco_dir <root> "
                f"--output_dir data/splits"
            )
    return split


def split_to_labeled_data(split: Dict,
                           label_dir: str = None) -> Dict:
    """
    Convert a split dict to a labeled_data dict for YOLOTrainer.

    Asymmetric protocol (Option A — frozen):
        image_list           ← labeled_train_ids       (90% — TRAINING input)
        inner_val_image_list ← labeled_inner_val_ids   (10% — YOLO early stopping)
        unlabeled_list       ← unlabeled_ids           (auto-labeling pool)

    The inner-val list is consumed by YOLOTrainer to populate
    {round_dir}/images/val/ and {round_dir}/labels/val/, frozen across all
    rounds, never receiving pseudo-labels.

    COCO note: labeled and unlabeled images both live in train2017/.
    The trainer handles this via list-level disjointness checking
    (not directory-level) when image_dir == unlabeled_image_dir.

    Args:
        split:     loaded split dict from load_split()
        label_dir: path to YOLO .txt label directory.
                   If None, defaults to {coco_root}/labels/train2017.
                   This directory MUST exist before training — run
                   convert_coco_to_yolo() first or pass --convert_labels.

    Raises:
        ValueError       if split is missing required train/inner_val fields
                         (regenerate with current make_splits.py).
        FileNotFoundError if label_dir does not exist.
    """
    # Required by Option A protocol — splits without these fields are
    # legacy and must be regenerated.
    for field in ("labeled_train_ids", "labeled_inner_val_ids"):
        if field not in split:
            raise ValueError(
                f"Split is missing required field {field!r}.\n"
                f"This split was generated before the Option A inner-val "
                f"protocol. Regenerate with current make_splits.py:\n"
                f"  python data/make_splits.py --coco_dir <root> "
                f"--output_dir data/splits"
            )

    train_dir              = Path(split["coco_train_dir"])
    labeled_train_ids      = sorted(set(split["labeled_train_ids"]))
    labeled_inner_val_ids  = sorted(set(split["labeled_inner_val_ids"]))
    unlabeled_ids          = sorted(set(split["unlabeled_ids"]))

    # COCO filenames are zero-padded 12-digit image IDs
    train_fnames     = [f"{img_id:012d}.jpg" for img_id in labeled_train_ids]
    inner_val_fnames = [f"{img_id:012d}.jpg" for img_id in labeled_inner_val_ids]
    unlabeled_fnames = [f"{img_id:012d}.jpg" for img_id in unlabeled_ids]

    # Resolve label_dir
    if label_dir is None:
        label_dir = str(train_dir.parent.parent / "labels" / "train2017")
    label_dir_path = Path(label_dir)

    if not label_dir_path.exists():
        raise FileNotFoundError(
            f"YOLO label directory not found: {label_dir_path}\n"
            f"Convert COCO annotations first:\n"
            f"  python data/make_splits.py \\\n"
            f"      --coco_dir <coco_root> \\\n"
            f"      --convert_labels \\\n"
            f"      --label_output_dir {label_dir_path}"
        )

    return {
        "image_dir":            str(train_dir),
        "label_dir":            str(label_dir_path),
        "image_list":           train_fnames,        # 90% — train input
        "inner_val_image_list": inner_val_fnames,    # 10% — frozen inner val
        "unlabeled_image_dir":  str(train_dir),      # same dir — list-level check
        "unlabeled_list":       unlabeled_fnames,
    }


# ── COCO → YOLO label conversion ─────────────────────────────────────────────

def convert_coco_to_yolo(annotation_file: str,
                          image_dir:       str,
                          output_label_dir: str,
                          image_ids:       List[int] = None) -> int:
    """
    Convert COCO instance annotations to YOLO .txt label format.

    Creates one .txt file per image in output_label_dir.
    File name: {image_id:012d}.txt
    Each line: <category_id_0indexed> <cx> <cy> <w> <h>  (normalised)

    Args:
        annotation_file:  path to instances_train2017.json
        image_dir:        path to image directory (for dimensions if needed)
        output_label_dir: directory to write .txt label files
        image_ids:        if provided, only convert these image IDs

    Returns:
        number of label files written
    """
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r") as f:
        data = json.load(f)

    # Build category ID → 0-indexed class map (sorted by category id)
    categories    = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_cls = {cat["id"]: i for i, cat in enumerate(categories)}
    n_classes     = len(categories)

    # Build image metadata lookup
    images_meta = {img["id"]: img for img in data["images"]}

    # Build annotation lookup: image_id → list of annotations
    ann_by_image: Dict[int, list] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    target_ids = set(image_ids) if image_ids else set(images_meta.keys())

    n_written = 0
    for img_id in sorted(target_ids):
        if img_id not in images_meta:
            continue

        meta = images_meta[img_id]
        w    = meta["width"]
        h    = meta["height"]

        anns  = ann_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]  # COCO: [x_top_left, y_top_left, w, h]
            if bw <= 0 or bh <= 0:
                continue
            cls = cat_id_to_cls.get(ann["category_id"])
            if cls is None:
                continue
            # Normalise
            cx  = max(0.0, min(1.0, (x + bw / 2) / w))
            cy  = max(0.0, min(1.0, (y + bh / 2) / h))
            nw  = max(0.0, min(1.0, bw / w))
            nh  = max(0.0, min(1.0, bh / h))
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out_path = output_label_dir / f"{img_id:012d}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_written += 1

    return n_written


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create COCO labeled/unlabeled splits for low-label experiments"
    )
    parser.add_argument(
        "--coco_dir", required=True,
        help="Root of COCO download (contains images/ and annotations/)")
    parser.add_argument(
        "--output_dir", default="data/splits",
        help="Directory to write split JSON files (default: data/splits)")
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.01, 0.05, 0.10],
        help="Label ratios (default: 0.01 0.05 0.10)")
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds per ratio (default: 5)")
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Base random seed (default: 42)")
    parser.add_argument(
        "--val_fraction", type=float, default=0.10,
        help="Fraction of labeled set held out as inner val for YOLO "
             "per-run early stopping (default: 0.10). The inner val is "
             "frozen across rounds and never receives pseudo-labels. It "
             "is NOT used by the outer-loop stopping rule.")
    parser.add_argument(
        "--convert_labels", action="store_true",
        help="Also convert COCO annotations to YOLO .txt format")
    parser.add_argument(
        "--label_output_dir", default=None,
        help="Where to write YOLO train .txt labels (required if --convert_labels)")
    parser.add_argument(
        "--convert_val_labels", action="store_true",
        help="Also convert val2017 annotations to YOLO .txt format")
    parser.add_argument(
        "--val_label_output_dir", default=None,
        help="Where to write YOLO val .txt labels (required if --convert_val_labels)")

    args = parser.parse_args()

    make_all_splits(
        coco_dir     = args.coco_dir,
        output_dir   = args.output_dir,
        ratios       = args.ratios,
        n_folds      = args.n_folds,
        base_seed    = args.base_seed,
        val_fraction = args.val_fraction,
    )

    if args.convert_labels:
        if not args.label_output_dir:
            parser.error("--label_output_dir required when --convert_labels is set")
        annotation_file = str(
            Path(args.coco_dir) / "annotations" / "instances_train2017.json")
        image_dir = str(
            Path(args.coco_dir) / "images" / "train2017")
        print(f"Converting train2017 annotations to YOLO format → {args.label_output_dir}")
        n = convert_coco_to_yolo(
            annotation_file  = annotation_file,
            image_dir        = image_dir,
            output_label_dir = args.label_output_dir,
        )
        print(f"Written {n:,} train YOLO label files")

    if args.convert_val_labels:
        if not args.val_label_output_dir:
            parser.error("--val_label_output_dir required when --convert_val_labels is set")
        val_annotation_file = str(
            Path(args.coco_dir) / "annotations" / "instances_val2017.json")
        val_image_dir = str(
            Path(args.coco_dir) / "images" / "val2017")
        print(f"Converting val2017 annotations to YOLO format → {args.val_label_output_dir}")
        n = convert_coco_to_yolo(
            annotation_file  = val_annotation_file,
            image_dir        = val_image_dir,
            output_label_dir = args.val_label_output_dir,
        )
        print(f"Written {n:,} val YOLO label files")