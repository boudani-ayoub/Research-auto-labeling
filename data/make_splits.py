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
    "ratio":           0.01,
    "fold":            0,
    "seed":            42,
    "n_labeled":       1182,
    "n_unlabeled":     117105,
    "n_val":           5000,
    "labeled_ids":     [int, ...],   -- COCO image_ids
    "unlabeled_ids":   [int, ...],   -- COCO image_ids
    "val_ids":         [int, ...],   -- COCO image_ids (from val2017)
    "coco_train_dir":  str,          -- absolute path to train2017 images
    "coco_val_dir":    str,          -- absolute path to val2017 images
    "annotation_file": str,          -- absolute path to instances_train2017.json
    "val_annotation_file": str,      -- absolute path to instances_val2017.json
    "created_at":      str,          -- ISO timestamp
  }

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
               val_annotation_file: str) -> Dict:
    """
    Create one labeled/unlabeled split.

    Args:
        all_train_ids:   sorted list of all train2017 image_ids
        val_ids:         sorted list of all val2017 image_ids
        ratio:           fraction of train2017 to use as labeled (e.g. 0.01)
        fold:            fold index 0..4
        base_seed:       base random seed
        coco_train_dir:  path to train2017 images directory
        coco_val_dir:    path to val2017 images directory
        annotation_file: path to instances_train2017.json
        val_annotation_file: path to instances_val2017.json

    Returns:
        split dict ready for JSON serialization
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

    return {
        "ratio":              ratio,
        "fold":               fold,
        "seed":               seed,
        "n_labeled":          len(labeled_ids),
        "n_unlabeled":        len(unlabeled_ids),
        "n_val":              len(val_ids),
        "labeled_ids":        labeled_ids,
        "unlabeled_ids":      unlabeled_ids,
        "val_ids":            val_ids,
        "coco_train_dir":     str(coco_train_dir),
        "coco_val_dir":       str(coco_val_dir),
        "annotation_file":    str(annotation_file),
        "val_annotation_file": str(val_annotation_file),
        "created_at":         datetime.now(timezone.utc).isoformat(),
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
                    base_seed:   int) -> None:
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
            )

            out_path = output_dir / f"coco_fold{fold}_{ratio_str}.json"
            with open(out_path, "w") as f:
                json.dump(split, f, indent=2)

            print(f"  fold {fold}: {split['n_labeled']:,} labeled, "
                  f"{split['n_unlabeled']:,} unlabeled → {out_path.name}")
            created.append(out_path)

        print()

    print(f"Created {len(created)} split files in {output_dir}/")
    return created


# ── Validation helpers (used by orchestrator and trainer) ────────────────────

def load_split(split_file: str) -> Dict:
    """Load a split JSON file. Validates required fields are present."""
    required = ["ratio", "fold", "labeled_ids", "unlabeled_ids",
                "coco_train_dir", "coco_val_dir", "annotation_file"]
    with open(split_file, "r") as f:
        split = json.load(f)
    for field in required:
        if field not in split:
            raise ValueError(
                f"Split file {split_file!r} is missing required field: {field!r}"
            )
    return split


def split_to_labeled_data(split: Dict,
                           label_dir: str = None) -> Dict:
    """
    Convert a split dict to a labeled_data dict for YOLOTrainer.

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
        FileNotFoundError if label_dir does not exist.
    """
    train_dir    = Path(split["coco_train_dir"])
    labeled_ids  = set(split["labeled_ids"])
    unlabeled_ids= set(split["unlabeled_ids"])

    # COCO filenames are zero-padded 12-digit image IDs
    labeled_fnames   = [f"{img_id:012d}.jpg" for img_id in sorted(labeled_ids)]
    unlabeled_fnames = [f"{img_id:012d}.jpg" for img_id in sorted(unlabeled_ids)]

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
        "image_dir":           str(train_dir),
        "label_dir":           str(label_dir_path),
        "image_list":          labeled_fnames,
        "unlabeled_image_dir": str(train_dir),   # same dir — trainer uses list-level check
        "unlabeled_list":      unlabeled_fnames,
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
        coco_dir   = args.coco_dir,
        output_dir = args.output_dir,
        ratios     = args.ratios,
        n_folds    = args.n_folds,
        base_seed  = args.base_seed,
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