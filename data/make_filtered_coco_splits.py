"""
data/make_filtered_coco_splits.py
==================================
Generate a class-filtered COCO 2017 split for the from-scratch
stability-gated iterative auto-labeling experiment.

Filters COCO train2017 + val2017 down to a selected set of vehicle
classes, remaps sparse COCO category IDs to dense YOLO IDs 0..K-1,
writes filtered YOLO-format label files, and emits a split JSON with
labeled / unlabeled / inner-val partitions.

Selected classes (locked for this experiment):
    bicycle    coco_id=2  → yolo_id=0
    car        coco_id=3  → yolo_id=1
    motorcycle coco_id=4  → yolo_id=2
    bus        coco_id=6  → yolo_id=3
    truck      coco_id=8  → yolo_id=4

Annotation filter (matches count_coco_classes.py exactly):
    skip if iscrowd == 1
    skip if bbox width <= 0 or height <= 0
    skip if category_id not in selected set

Image filter:
    only keep images with >= 1 valid selected annotation
    (drops images that end up empty after filtering — no
    "background" training images for this experiment)

Inner-val protocol (matches make_splits.py):
    base_seed: fold_seed(base, fold)        = base + fold*1000
    inner-val: inner_val_seed(base, fold)   = fold_seed + 500
    val_fraction default 0.10 (90/10 train/inner-val carve-out)

Usage:
    python data/make_filtered_coco_splits.py \\
        --coco_dir   ~/research/coco \\
        --output_dir ~/research/coco_filtered_vehicles \\
        --splits_dir data/splits_vehicles \\
        --ratios     0.05 0.10 0.20 \\
        --n_folds    5 \\
        --base_seed  42

Output:
    ~/research/coco_filtered_vehicles/
        images/
            train2017/{image_id:012d}.jpg   (SYMLINKS to original COCO images)
            val2017/{image_id:012d}.jpg     (SYMLINKS to original COCO images)
        labels/
            train2017/{image_id:012d}.txt   (filtered + remapped YOLO labels)
            val2017/{image_id:012d}.txt
        train_image_list.txt                (abs paths into images/train2017/)
        val_image_list.txt                  (abs paths into images/val2017/)
        filter_stats.json                   (audit numbers)

    data/splits_vehicles/
        coco_vehicles_fold{0..4}_{1pct,5pct,10pct,20pct}.json

Why symlinks under images/ inside the output root:
    Ultralytics infers label paths from image paths by substituting
    '/images/' with '/labels/' and changing the file extension to .txt.
    If image paths point at the original ~/research/coco/images/ tree,
    YOLO will look for labels at ~/research/coco/labels/, where stale
    80-class label files from a previous full-COCO run may still live.
    Symlinks under <output_root>/images/ ensure the path substitution
    finds only the filtered 5-class .txt files.

DEFERRED — must be done before training runs:
    1. A `filtered_split_to_labeled_data()` helper in data/make_splits.py
       that consumes this script's split JSON and returns a labeled_data
       dict using image_dir_train (the symlink dir), label_dir_train,
       and labeled_train_ids / labeled_inner_val_ids / unlabeled_ids.

    2. Verifier script `verify_filtered_coco.py` (Step 3 of the plan).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ── Locked class set ─────────────────────────────────────────────────────────
# This is hardcoded by design. Don't make it a CLI argument — the entire
# experiment protocol depends on these exact 5 classes and their dense IDs.
# Confirmed viable via count_coco_classes.py (all >= 500 instances at 10%).

CLASS_REMAP: Dict[int, int] = {
    2: 0,   # bicycle
    3: 1,   # car
    4: 2,   # motorcycle
    6: 3,   # bus
    8: 4,   # truck
}

CLASS_NAMES: List[str] = ["bicycle", "car", "motorcycle", "bus", "truck"]

SELECTED_COCO_IDS: Set[int] = set(CLASS_REMAP.keys())


# ── Seed helpers (mirror make_splits.py exactly) ─────────────────────────────

def fold_seed(base_seed: int, fold: int) -> int:
    return base_seed + fold * 1000


def inner_val_seed(base_seed: int, fold: int) -> int:
    return fold_seed(base_seed, fold) + 500


# ── Ratio formatting ─────────────────────────────────────────────────────────

def ratio_to_str(r: float) -> str:
    """0.10 → '10pct', 0.05 → '5pct', 0.01 → '1pct'."""
    pct = round(r * 100)
    return f"{pct}pct"


# ── COCO loader and filter ───────────────────────────────────────────────────

def load_and_filter_coco(ann_file: Path
                          ) -> Tuple[Dict[int, Dict],
                                     Dict[int, List[Dict]],
                                     Dict[str, int]]:
    """
    Read a COCO annotations JSON. Apply iscrowd / invalid-bbox / class
    filters. Return:
        images_meta:       image_id → {file_name, width, height}
                           (all images in source JSON, even ones that get
                            dropped by the empty-image filter later)
        annotations_kept:  image_id → list of valid filtered annotations
                           (only images with ≥1 valid annotation appear)
        skip_stats:        audit counts
    """
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    print(f"  Loading {ann_file.name} ... ", end="", flush=True)
    with open(ann_file, "r") as f:
        data = json.load(f)
    print("done.")

    # Index image metadata by image_id.
    images_meta: Dict[int, Dict] = {
        img["id"]: {
            "file_name": img["file_name"],
            "width":     img["width"],
            "height":    img["height"],
        }
        for img in data.get("images", [])
    }

    annotations_kept: Dict[int, List[Dict]] = {}
    skip = {
        "skipped_iscrowd":         0,
        "skipped_invalid_bbox":    0,
        "skipped_unselected_class": 0,
        "kept":                    0,
    }

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            skip["skipped_iscrowd"] += 1
            continue

        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            skip["skipped_invalid_bbox"] += 1
            continue
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            skip["skipped_invalid_bbox"] += 1
            continue

        cat_id = ann["category_id"]
        if cat_id not in SELECTED_COCO_IDS:
            skip["skipped_unselected_class"] += 1
            continue

        annotations_kept.setdefault(ann["image_id"], []).append({
            "category_id": cat_id,
            "bbox":        (x, y, w, h),
        })
        skip["kept"] += 1

    return images_meta, annotations_kept, skip


# ── YOLO label conversion ────────────────────────────────────────────────────

def coco_bbox_to_yolo(x: float, y: float, w: float, h: float,
                       img_w: int, img_h: int
                       ) -> "Tuple[float, float, float, float] | None":
    """
    Convert COCO bbox (x_top_left, y_top_left, w, h) in absolute pixels
    to YOLO bbox (cx, cy, w, h) normalized to [0, 1].

    Clipping is done in CORNER SPACE before deriving (cx, cy, w, h).
    Independent clamping on cx/cy/w/h after normalization is mathematically
    wrong — it can distort boxes (e.g. a small box partially off the image
    becomes a huge box hugging the image corner). The correct procedure:

        1. Compute corners (x1, y1, x2, y2) from (x, y, w, h).
        2. Clip each corner to [0, img_w] or [0, img_h].
        3. If the clipped region has zero or negative area, the box is
           entirely outside the image — return None and the caller drops it.
        4. Otherwise derive cx, cy, w, h from the clipped corners.

    Returns:
        (cx, cy, nw, nh) all in [0, 1] on success.
        None if the clipped box has zero area (annotation is unusable).
    """
    iw = float(img_w)
    ih = float(img_h)

    x1 = max(0.0, min(iw, x))
    y1 = max(0.0, min(ih, y))
    x2 = max(0.0, min(iw, x + w))
    y2 = max(0.0, min(ih, y + h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None

    cx = (x1 + x2) / 2.0 / iw
    cy = (y1 + y2) / 2.0 / ih
    nw = bw / iw
    nh = bh / ih

    return cx, cy, nw, nh


def write_yolo_labels(annotations_kept: Dict[int, List[Dict]],
                       images_meta:      Dict[int, Dict],
                       out_label_dir:    Path) -> Tuple[Dict[str, int], Set[int]]:
    """
    Write one .txt per image with valid annotations. File name is the
    image_id zero-padded to 12 digits (matching COCO convention).

    Each label line: "<yolo_id> <cx> <cy> <w> <h>" with 6-decimal precision.
    Written in deterministic per-image order (sorted by yolo_id, then by
    cx, cy) so re-running produces byte-identical files.

    Annotations may be dropped here for two reasons:
      - degenerate bbox after corner-clipping (zero/negative area)
      - no valid annotations remain → image .txt is not written

    Returns:
        stats:           audit dict
        written_ids:     set of image_ids that received a non-empty .txt file
                         (used downstream as the canonical filtered set —
                         load_and_filter_coco returns image_ids that *had*
                         a valid annotation in the source JSON, but post-clip
                         a few may be lost.)
    """
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "label_files_written":           0,
        "label_lines_written":           0,
        "labels_clipped":                0,
        "labels_dropped_zero_area":      0,
        "label_files_skipped_no_dims":   0,
        "label_files_skipped_no_anns_left": 0,
    }
    written_ids: Set[int] = set()

    for image_id in sorted(annotations_kept.keys()):
        meta = images_meta.get(image_id)
        if meta is None or meta["width"] <= 0 or meta["height"] <= 0:
            stats["label_files_skipped_no_dims"] += 1
            continue

        img_w = meta["width"]
        img_h = meta["height"]

        lines: List[Tuple[int, float, float, float, float]] = []
        for ann in annotations_kept[image_id]:
            yolo_id = CLASS_REMAP[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            corner_clipped = (
                x < 0 or y < 0 or
                (x + w) > img_w or (y + h) > img_h
            )

            result = coco_bbox_to_yolo(x, y, w, h, img_w, img_h)
            if result is None:
                stats["labels_dropped_zero_area"] += 1
                continue
            cx, cy, nw, nh = result

            if corner_clipped:
                stats["labels_clipped"] += 1

            lines.append((yolo_id, cx, cy, nw, nh))

        if not lines:
            stats["label_files_skipped_no_anns_left"] += 1
            continue

        lines.sort(key=lambda t: (t[0], t[1], t[2]))

        out_path = out_label_dir / f"{image_id:012d}.txt"
        with open(out_path, "w") as f:
            for yolo_id, cx, cy, nw, nh in lines:
                f.write(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        stats["label_files_written"] += 1
        stats["label_lines_written"] += len(lines)
        written_ids.add(image_id)

    return stats, written_ids


# ── Inner-val sub-split ──────────────────────────────────────────────────────

def split_labeled_into_train_val(labeled_ids:  List[int],
                                  base_seed:    int,
                                  fold:         int,
                                  val_fraction: float = 0.10
                                  ) -> Tuple[List[int], List[int]]:
    """
    Same logic as make_splits.py.split_labeled_into_train_val.
    Independent RNG seeded by inner_val_seed(base, fold). Input order
    independent (canonical sort applied before shuffle). Deterministic.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if len(labeled_ids) < 10:
        raise ValueError(
            f"labeled_ids has only {len(labeled_ids)} items; refusing to split "
            f"into train/inner-val (need at least 10)."
        )

    n_total = len(labeled_ids)
    n_val   = max(1, round(n_total * val_fraction))

    canonical = sorted(labeled_ids)
    shuffled  = canonical.copy()
    rng       = random.Random(inner_val_seed(base_seed, fold))
    rng.shuffle(shuffled)

    inner_val_ids = sorted(shuffled[:n_val])
    train_ids     = sorted(shuffled[n_val:])

    assert set(train_ids).isdisjoint(set(inner_val_ids))
    assert set(train_ids) | set(inner_val_ids) == set(labeled_ids)

    return train_ids, inner_val_ids


def symlink_filtered_images(image_ids:        List[int],
                              src_image_dir:   Path,
                              dst_image_dir:   Path) -> int:
    """
    Create a symlink under dst_image_dir for each image_id in the list,
    pointing to the corresponding file in src_image_dir.

    Why symlinks (not copies, not raw image-list files alone):
        Ultralytics derives label paths from image paths by replacing
        '/images/' with '/labels/' and changing the extension to .txt.
        If image paths point at the original COCO directory (which may
        have stale 80-class label files in a parallel /labels/ tree),
        YOLO will pick up the wrong labels and silently train/evaluate
        on the wrong supervision.

        Symlinks under <output_root>/images/{train,val}2017/ make the
        path substitution land inside the filtered output tree, where
        only the filtered 5-class .txt files live.

    Returns:
        number of symlinks successfully created.

    The destination directory is assumed already cleared by the caller.
    Existing symlinks at the destination are overwritten.
    """
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    n_made = 0
    for image_id in image_ids:
        fname = f"{image_id:012d}.jpg"
        src   = src_image_dir / fname
        dst   = dst_image_dir / fname
        if not src.exists():
            # Source image missing — skip rather than fail loudly. The
            # verifier will catch this as part of its sanity checks.
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        n_made += 1
    return n_made


# ── Split generation ─────────────────────────────────────────────────────────

def make_one_split(filtered_train_ids:  List[int],
                    filtered_val_ids:   List[int],
                    ratio:              float,
                    fold:               int,
                    base_seed:          int,
                    coco_train_dir:     str,
                    coco_val_dir:       str,
                    annotation_file:    str,
                    val_annotation_file: str,
                    label_dir_train:    str,
                    label_dir_val:      str,
                    image_dir_train:    str,
                    image_dir_val:      str,
                    train_image_list_path: str,
                    val_image_list_path:   str,
                    val_fraction:       float = 0.10) -> Dict:
    """
    Build one split dict for (ratio, fold) over the filtered image pool.
    """
    seed = fold_seed(base_seed, fold)
    rng  = random.Random(seed)

    n_total   = len(filtered_train_ids)
    n_labeled = max(1, round(n_total * ratio))

    shuffled = sorted(filtered_train_ids)  # canonical order before shuffle
    rng.shuffle(shuffled)

    labeled_ids   = sorted(shuffled[:n_labeled])
    unlabeled_ids = sorted(shuffled[n_labeled:])

    labeled_train_ids, labeled_inner_val_ids = split_labeled_into_train_val(
        labeled_ids  = labeled_ids,
        base_seed    = base_seed,
        fold         = fold,
        val_fraction = val_fraction,
    )

    return {
        # Existing schema (matches make_splits.py)
        "ratio":                   ratio,
        "fold":                    fold,
        "seed":                    seed,
        "inner_val_seed":          inner_val_seed(base_seed, fold),
        "val_fraction":            val_fraction,
        "n_labeled":               len(labeled_ids),
        "n_labeled_train":         len(labeled_train_ids),
        "n_labeled_inner_val":     len(labeled_inner_val_ids),
        "n_unlabeled":             len(unlabeled_ids),
        "n_val":                   len(filtered_val_ids),
        "labeled_ids":             labeled_ids,
        "labeled_train_ids":       labeled_train_ids,
        "labeled_inner_val_ids":   labeled_inner_val_ids,
        "unlabeled_ids":           unlabeled_ids,
        "val_ids":                 filtered_val_ids,
        "coco_train_dir":          coco_train_dir,
        "coco_val_dir":            coco_val_dir,
        "annotation_file":         annotation_file,
        "val_annotation_file":     val_annotation_file,
        "created_at":              datetime.now(timezone.utc).isoformat(),

        # New fields specific to filtered-class experiment
        "class_filter":            sorted(SELECTED_COCO_IDS),
        "class_remap":             {str(k): v for k, v in CLASS_REMAP.items()},
        "class_names":             CLASS_NAMES,
        # image_dir_{train,val} point at SYMLINK directories under the
        # filtered-output tree. Use these (not coco_train_dir/coco_val_dir)
        # for any reading that involves Ultralytics' /images/→/labels/
        # substitution. The original coco_*_dir paths are retained above
        # only for provenance / audit.
        "image_dir_train":         image_dir_train,
        "image_dir_val":           image_dir_val,
        "label_dir_train":         label_dir_train,
        "label_dir_val":           label_dir_val,
        "train_image_list_path":   train_image_list_path,
        "val_image_list_path":     val_image_list_path,
        "filter_description":      "vehicle-only (bicycle, car, motorcycle, bus, truck)",
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate filtered-COCO splits for vehicle-only experiment"
    )
    parser.add_argument("--coco_dir",   required=True, type=Path,
        help="Path to COCO root (containing images/ and annotations/)")
    parser.add_argument("--output_dir", required=True, type=Path,
        help="Where to write filtered YOLO labels and filter_stats.json")
    parser.add_argument("--splits_dir", required=True, type=Path,
        help="Where to write split JSON files")
    parser.add_argument("--ratios", type=float, nargs="+",
        default=[0.05, 0.10, 0.20],
        help="Labeled ratios (default: 0.05 0.10 0.20)")
    parser.add_argument("--n_folds", type=int, default=5,
        help="Number of folds (default: 5)")
    parser.add_argument("--base_seed", type=int, default=42,
        help="Base random seed (default: 42)")
    parser.add_argument("--val_fraction", type=float, default=0.10,
        help="Inner-val fraction within labeled pool (default: 0.10)")
    args = parser.parse_args()

    coco_dir   = args.coco_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()

    train_image_dir  = coco_dir / "images" / "train2017"
    val_image_dir    = coco_dir / "images" / "val2017"
    train_ann_file   = coco_dir / "annotations" / "instances_train2017.json"
    val_ann_file     = coco_dir / "annotations" / "instances_val2017.json"

    for p, name in [(train_image_dir, "train2017 images"),
                    (val_image_dir,   "val2017 images"),
                    (train_ann_file,  "instances_train2017.json"),
                    (val_ann_file,    "instances_val2017.json")]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p} ({name})")

    label_dir_train = output_dir / "labels" / "train2017"
    label_dir_val   = output_dir / "labels" / "val2017"
    image_dir_train_out = output_dir / "images" / "train2017"
    image_dir_val_out   = output_dir / "images" / "val2017"

    print()
    print("=" * 76)
    print("  Filtered-COCO split generator (vehicle-only)")
    print("=" * 76)
    print(f"  COCO root:    {coco_dir}")
    print(f"  Output root:  {output_dir}")
    print(f"  Splits out:   {splits_dir}")
    print(f"  Classes:      {CLASS_NAMES}")
    print(f"  Remap:        " + ", ".join(
        f"{k}→{v}" for k, v in sorted(CLASS_REMAP.items())))
    print(f"  Ratios:       {args.ratios}")
    print(f"  Folds:        {args.n_folds}")
    print(f"  base_seed:    {args.base_seed}")
    print(f"  val_fraction: {args.val_fraction}")
    print()

    # ── Clear stale output (safety against re-runs with changed filters) ─────
    # If we leave old .txt or symlinks from a previous run, they may not match
    # the current filter, contaminating the dataset.
    for d in (label_dir_train, label_dir_val,
              image_dir_train_out, image_dir_val_out):
        if d.exists():
            print(f"  Clearing stale output: {d}")
            shutil.rmtree(d)
    print()

    # ── Filter and convert train2017 ─────────────────────────────────────────
    print("Filtering train2017 ...")
    train_meta, train_anns, train_skip = load_and_filter_coco(train_ann_file)
    train_label_stats, train_written_ids = write_yolo_labels(
        train_anns, train_meta, label_dir_train
    )
    filtered_train_ids = sorted(train_written_ids)
    print(f"  filtered train images:  {len(filtered_train_ids):,}")
    print(f"  label files written:    {train_label_stats['label_files_written']:,}")
    print(f"  label lines written:    {train_label_stats['label_lines_written']:,}")
    if train_label_stats['labels_clipped']:
        print(f"  labels corner-clipped:  {train_label_stats['labels_clipped']:,}")
    if train_label_stats['labels_dropped_zero_area']:
        print(f"  labels dropped (zero area after clip): "
              f"{train_label_stats['labels_dropped_zero_area']:,}")
    if train_label_stats['label_files_skipped_no_anns_left']:
        print(f"  images dropped (no anns survived clip): "
              f"{train_label_stats['label_files_skipped_no_anns_left']:,}")

    # Create image symlinks under the filtered-output tree so Ultralytics'
    # path substitution (/images/ → /labels/) lands in the right place.
    n_train_links = symlink_filtered_images(
        filtered_train_ids, train_image_dir, image_dir_train_out
    )
    print(f"  train image symlinks:   {n_train_links:,}")
    if n_train_links != len(filtered_train_ids):
        print(f"  WARNING: {len(filtered_train_ids) - n_train_links} train "
              f"image(s) missing from {train_image_dir}")
    print()

    # ── Filter and convert val2017 ───────────────────────────────────────────
    print("Filtering val2017 ...")
    val_meta, val_anns, val_skip = load_and_filter_coco(val_ann_file)
    val_label_stats, val_written_ids = write_yolo_labels(
        val_anns, val_meta, label_dir_val
    )
    filtered_val_ids = sorted(val_written_ids)
    print(f"  filtered val images:    {len(filtered_val_ids):,}")
    print(f"  label files written:    {val_label_stats['label_files_written']:,}")
    print(f"  label lines written:    {val_label_stats['label_lines_written']:,}")
    if val_label_stats['labels_clipped']:
        print(f"  labels corner-clipped:  {val_label_stats['labels_clipped']:,}")
    if val_label_stats['labels_dropped_zero_area']:
        print(f"  labels dropped (zero area after clip): "
              f"{val_label_stats['labels_dropped_zero_area']:,}")

    n_val_links = symlink_filtered_images(
        filtered_val_ids, val_image_dir, image_dir_val_out
    )
    print(f"  val image symlinks:     {n_val_links:,}")
    if n_val_links != len(filtered_val_ids):
        print(f"  WARNING: {len(filtered_val_ids) - n_val_links} val "
              f"image(s) missing from {val_image_dir}")
    print()

    # Sanity invariant: every class must have >=1 image in filtered val.
    val_class_presence: Dict[int, int] = {yid: 0 for yid in CLASS_REMAP.values()}
    for image_id in filtered_val_ids:
        for ann in val_anns[image_id]:
            meta = val_meta[image_id]
            if coco_bbox_to_yolo(*ann["bbox"], meta["width"], meta["height"]) is None:
                continue
            val_class_presence[CLASS_REMAP[ann["category_id"]]] += 1
    missing = [CLASS_NAMES[yid] for yid, c in val_class_presence.items() if c == 0]
    if missing:
        print(f"  ERROR: filtered val2017 has zero instances for class(es): {missing}")
        print(f"  This will break per-class mAP evaluation. Aborting.")
        return 1

    # ── Write image-list files pointing at the SYMLINKED paths ──────────────
    # Critical: paths must live under output_dir/images/ so Ultralytics'
    # /images/→/labels/ substitution finds the filtered .txt files.
    train_list_path = output_dir / "train_image_list.txt"
    val_list_path   = output_dir / "val_image_list.txt"
    with open(train_list_path, "w") as f:
        for image_id in filtered_train_ids:
            f.write(f"{image_dir_train_out / f'{image_id:012d}.jpg'}\n")
    with open(val_list_path, "w") as f:
        for image_id in filtered_val_ids:
            f.write(f"{image_dir_val_out / f'{image_id:012d}.jpg'}\n")
    print(f"  Train image list (filtered): {train_list_path}")
    print(f"  Val   image list (filtered): {val_list_path}")
    print()

    # ── Write filter audit JSON ──────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_stats = {
        "selected_coco_ids":     sorted(SELECTED_COCO_IDS),
        "class_remap":           {str(k): v for k, v in CLASS_REMAP.items()},
        "class_names":           CLASS_NAMES,
        "train": {
            "skip":                train_skip,
            "label_files_written": train_label_stats["label_files_written"],
            "label_lines_written": train_label_stats["label_lines_written"],
            "labels_clipped":      train_label_stats["labels_clipped"],
            "labels_dropped_zero_area":         train_label_stats["labels_dropped_zero_area"],
            "label_files_skipped_no_anns_left": train_label_stats["label_files_skipped_no_anns_left"],
            "filtered_image_ids":  len(filtered_train_ids),
        },
        "val": {
            "skip":                val_skip,
            "label_files_written": val_label_stats["label_files_written"],
            "label_lines_written": val_label_stats["label_lines_written"],
            "labels_clipped":      val_label_stats["labels_clipped"],
            "labels_dropped_zero_area":         val_label_stats["labels_dropped_zero_area"],
            "label_files_skipped_no_anns_left": val_label_stats["label_files_skipped_no_anns_left"],
            "filtered_image_ids":  len(filtered_val_ids),
        },
        "val_class_presence":    {CLASS_NAMES[yid]: c
                                   for yid, c in val_class_presence.items()},
        "train_image_list_path": str(train_list_path),
        "val_image_list_path":   str(val_list_path),
        "image_dir_train":       str(image_dir_train_out),
        "image_dir_val":         str(image_dir_val_out),
        "label_dir_train":       str(label_dir_train),
        "label_dir_val":         str(label_dir_val),
        "created_at":            datetime.now(timezone.utc).isoformat(),
    }
    stats_path = output_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(filter_stats, f, indent=2)
    print(f"  Filter audit written: {stats_path}")
    print()

    # ── Generate splits ──────────────────────────────────────────────────────
    splits_dir.mkdir(parents=True, exist_ok=True)
    print("Generating splits ...")
    n_created = 0
    for ratio in args.ratios:
        ratio_str = ratio_to_str(ratio)
        for fold in range(args.n_folds):
            split = make_one_split(
                filtered_train_ids  = filtered_train_ids,
                filtered_val_ids    = filtered_val_ids,
                ratio               = ratio,
                fold                = fold,
                base_seed           = args.base_seed,
                coco_train_dir      = str(train_image_dir),
                coco_val_dir        = str(val_image_dir),
                annotation_file     = str(train_ann_file),
                val_annotation_file = str(val_ann_file),
                label_dir_train     = str(label_dir_train),
                label_dir_val       = str(label_dir_val),
                image_dir_train     = str(image_dir_train_out),
                image_dir_val       = str(image_dir_val_out),
                train_image_list_path = str(train_list_path),
                val_image_list_path   = str(val_list_path),
                val_fraction        = args.val_fraction,
            )
            out_path = splits_dir / f"coco_vehicles_fold{fold}_{ratio_str}.json"
            with open(out_path, "w") as f:
                json.dump(split, f, indent=2)
            print(f"  fold {fold} {ratio_str}: "
                  f"{split['n_labeled_train']:,} train + "
                  f"{split['n_labeled_inner_val']:,} inner_val + "
                  f"{split['n_unlabeled']:,} unlabeled "
                  f"→ {out_path.name}")
            n_created += 1

    print()
    print(f"Created {n_created} split files in {splits_dir}/")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())