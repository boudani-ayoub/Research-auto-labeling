"""
data/count_coco_classes.py
==========================
Count instances and images for candidate class subsets in COCO 2017.

Purpose:
    Before locking the class subset for the filtered-COCO experiment,
    we need to verify each candidate class has enough instances at the
    target labeled ratio (10%) to give a defensible training signal.

    A class with ~50 instances at 10% labeled is unworkable — the
    model will essentially never see it.

Outputs:
    For each candidate class, prints:
        - instances in train2017
        - images containing it in train2017
        - instances in val2017
        - images containing it in val2017
        - estimated instances at 10% labeled (= train2017_instances * 0.1)

    Two candidate sets are reported side by side for comparison:
        Default (vehicle-only):
            bicycle, car, motorcycle, bus, truck
        Alternative (mixed-domain):
            person, bicycle, car, dog, chair

Usage:
    cd ~/research/stability_autolabel
    python data/count_coco_classes.py \\
        --train_ann ~/research/coco/annotations/instances_train2017.json \\
        --val_ann   ~/research/coco/annotations/instances_val2017.json

    Optional: --labeled_ratio 0.10 (default 0.10, override for sensitivity check)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ── Candidate class sets (COCO category_id values from the official taxonomy) ─
# COCO category_id values:
#     1=person, 2=bicycle, 3=car, 4=motorcycle, 6=bus, 8=truck,
#     18=dog, 62=chair
#
# Note: COCO category_id values are SPARSE / non-contiguous.
# IDs in COCO 80-class detection skip values: e.g., 12, 26, 29, 30, 45, 66,
# 68, 69, 71, 83 are unused. So `category_id - 1` is NOT a valid mapping
# to YOLO class IDs.
#
# For the filtered experiment, the next script must build an EXPLICIT dense
# remap from the selected sparse COCO IDs to 0..K-1, e.g.:
#     bicycle    2 → 0
#     car        3 → 1
#     motorcycle 4 → 2
#     bus        6 → 3
#     truck      8 → 4
#
# This count script reports the original sparse COCO IDs only.

VEHICLE_CLASSES: Dict[str, int] = {
    "bicycle":    2,
    "car":        3,
    "motorcycle": 4,
    "bus":        6,
    "truck":      8,
}

MIXED_CLASSES: Dict[str, int] = {
    "person":   1,
    "bicycle":  2,
    "car":      3,
    "dog":     18,
    "chair":   62,
}


def load_coco_counts(ann_file: Path,
                      class_ids: Set[int]
                      ) -> Tuple[Dict[int, int], Dict[int, Set[int]], Dict[str, int]]:
    """
    Read a COCO annotations JSON and return:
        instance_counts:  category_id → number of valid instance annotations
        image_id_sets:    category_id → set of image_ids containing that class
        skip_stats:       summary of annotations skipped (by reason)

    Only categories in class_ids are tracked. Other categories are ignored.

    Annotations are skipped under these conditions to match the eventual
    label-conversion behavior in make_filtered_coco_splits.py:
      - iscrowd == 1 (crowd masks, not instance annotations — YOLO drops these)
      - bbox width <= 0 or height <= 0 (corrupt boxes; a small number exist
        in real COCO; well-documented)

    If we counted these here but the conversion step drops them, our
    viability estimates would be inflated.
    """
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    print(f"  Loading {ann_file.name} ... ", end="", flush=True)
    with open(ann_file, "r") as f:
        data = json.load(f)
    print("done.")

    instance_counts: Dict[int, int] = defaultdict(int)
    image_id_sets:   Dict[int, Set[int]] = defaultdict(set)
    skip_stats = {
        "skipped_iscrowd":      0,
        "skipped_invalid_bbox": 0,
        "kept_in_class_set":    0,
        "kept_outside_class_set": 0,
    }

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            skip_stats["skipped_iscrowd"] += 1
            continue

        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            skip_stats["skipped_invalid_bbox"] += 1
            continue
        _, _, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            skip_stats["skipped_invalid_bbox"] += 1
            continue

        cat_id = ann["category_id"]
        if cat_id in class_ids:
            instance_counts[cat_id] += 1
            image_id_sets[cat_id].add(ann["image_id"])
            skip_stats["kept_in_class_set"] += 1
        else:
            skip_stats["kept_outside_class_set"] += 1

    return dict(instance_counts), dict(image_id_sets), skip_stats


def load_official_category_names(ann_file: Path) -> Dict[int, str]:
    """Read category_id → category name mapping from the COCO JSON."""
    with open(ann_file, "r") as f:
        data = json.load(f)
    return {c["id"]: c["name"] for c in data.get("categories", [])}


def report_class_set(set_name:        str,
                      classes:        Dict[str, int],
                      train_counts:   Dict[int, int],
                      train_image_sets: Dict[int, Set[int]],
                      val_counts:     Dict[int, int],
                      val_image_sets: Dict[int, Set[int]],
                      labeled_ratio:  float,
                      official_names: Dict[int, str]) -> None:
    """Print a side-by-side report for one candidate class set."""

    print()
    print("=" * 88)
    print(f"  {set_name}")
    print("=" * 88)
    print(f"  {'class':<14}{'coco_id':>8}  "
          f"{'train_inst':>11}  {'train_imgs':>11}  "
          f"{'val_inst':>9}  {'val_imgs':>9}  "
          f"{'~at_'+f'{int(labeled_ratio*100)}%':>10}")
    print("  " + "-" * 86)

    # Track totals + cross-check that names are correct.
    total_train_inst = 0
    total_train_img_union: Set[int] = set()
    issues: List[str] = []

    for name, coco_id in classes.items():
        # Sanity: name in our candidate must match the official name in COCO.
        official = official_names.get(coco_id, "<missing>")
        if official != name:
            issues.append(
                f"NAME MISMATCH for coco_id={coco_id}: "
                f"we expect '{name}' but COCO says '{official}'"
            )

        t_inst   = train_counts.get(coco_id, 0)
        t_imgs   = len(train_image_sets.get(coco_id, set()))
        v_inst   = val_counts.get(coco_id, 0)
        v_imgs   = len(val_image_sets.get(coco_id, set()))
        est_at_r = int(t_inst * labeled_ratio)

        total_train_inst += t_inst
        total_train_img_union.update(train_image_sets.get(coco_id, set()))

        print(f"  {name:<14}{coco_id:>8}  "
              f"{t_inst:>11,}  {t_imgs:>11,}  "
              f"{v_inst:>9,}  {v_imgs:>9,}  "
              f"{est_at_r:>10,}")

    print("  " + "-" * 86)
    print(f"  {'union':<14}{'-':>8}  "
          f"{total_train_inst:>11,}  {len(total_train_img_union):>11,}  "
          f"{'-':>9}  {'-':>9}  "
          f"{int(total_train_inst * labeled_ratio):>10,}")
    print()
    print(f"  total filtered train images (≥1 selected class): "
          f"{len(total_train_img_union):,}")
    print(f"  estimated labeled-pool images at {int(labeled_ratio*100)}% "
          f"(before inner-val carve-out): "
          f"~{int(len(total_train_img_union) * labeled_ratio):,}")

    if issues:
        print()
        print("  WARNINGS:")
        for issue in issues:
            print(f"    - {issue}")


def viability_summary(classes:        Dict[str, int],
                       train_counts:   Dict[int, int],
                       labeled_ratio:  float,
                       min_inst_threshold: int = 500) -> List[str]:
    """
    Return a list of human-readable viability warnings for a class set.

    Heuristic (not a hard rule):
        At 10% labeled, fewer than ~500 instances per class is borderline.
        The model will see only ~50 examples per epoch; YOLO needs more
        to learn a class reliably.
    """
    warnings: List[str] = []
    for name, coco_id in classes.items():
        t_inst   = train_counts.get(coco_id, 0)
        est_at_r = int(t_inst * labeled_ratio)
        if est_at_r < min_inst_threshold:
            warnings.append(
                f"'{name}' has only ~{est_at_r:,} instances at "
                f"{int(labeled_ratio*100)}% labeled "
                f"(below threshold of {min_inst_threshold:,}). "
                f"Class may not learn reliably."
            )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count COCO 2017 instances/images for candidate class sets"
    )
    parser.add_argument(
        "--train_ann", required=True, type=Path,
        help="Path to instances_train2017.json"
    )
    parser.add_argument(
        "--val_ann", required=True, type=Path,
        help="Path to instances_val2017.json"
    )
    parser.add_argument(
        "--labeled_ratio", type=float, default=0.10,
        help="Labeled ratio for viability estimate (default 0.10)"
    )
    parser.add_argument(
        "--min_inst_threshold", type=int, default=500,
        help="Below this many instances at the labeled ratio, the class is "
             "flagged as borderline (default 500)"
    )
    args = parser.parse_args()

    # Build the union of all class IDs we want to count.
    all_class_ids: Set[int] = set(VEHICLE_CLASSES.values()) | set(MIXED_CLASSES.values())

    print()
    print("Loading COCO 2017 annotations...")
    print()

    train_inst, train_imgs, train_skips = load_coco_counts(args.train_ann, all_class_ids)
    val_inst,   val_imgs,   val_skips   = load_coco_counts(args.val_ann,   all_class_ids)

    # Cross-check class names against COCO's official taxonomy.
    official_names = load_official_category_names(args.train_ann)

    # Report each set separately.
    report_class_set(
        set_name        = "DEFAULT — vehicle-only",
        classes         = VEHICLE_CLASSES,
        train_counts    = train_inst,
        train_image_sets= train_imgs,
        val_counts      = val_inst,
        val_image_sets  = val_imgs,
        labeled_ratio   = args.labeled_ratio,
        official_names  = official_names,
    )

    report_class_set(
        set_name        = "ALTERNATIVE — mixed-domain (for comparison only)",
        classes         = MIXED_CLASSES,
        train_counts    = train_inst,
        train_image_sets= train_imgs,
        val_counts      = val_inst,
        val_image_sets  = val_imgs,
        labeled_ratio   = args.labeled_ratio,
        official_names  = official_names,
    )

    # Annotation-level skip report — confirms we filtered iscrowd / invalid bboxes
    # the same way the eventual label converter will.
    print()
    print("=" * 88)
    print("  Annotation filter audit (matches label-converter behavior)")
    print("=" * 88)
    for label, stats in [("train2017", train_skips), ("val2017", val_skips)]:
        total = sum(stats.values())
        print(f"  {label}: total annotations = {total:,}")
        print(f"    skipped iscrowd       = {stats['skipped_iscrowd']:,}")
        print(f"    skipped invalid bbox  = {stats['skipped_invalid_bbox']:,}")
        print(f"    kept (selected sets)  = {stats['kept_in_class_set']:,}")
        print(f"    kept (other classes)  = {stats['kept_outside_class_set']:,}")

    # Viability summary for the default set.
    print()
    print("=" * 88)
    print("  Viability summary (default set)")
    print("=" * 88)
    warns = viability_summary(VEHICLE_CLASSES, train_inst, args.labeled_ratio,
                               args.min_inst_threshold)
    if not warns:
        print(f"  OK — every class in the vehicle set has at least "
              f"{args.min_inst_threshold:,} estimated instances at "
              f"{int(args.labeled_ratio*100)}% labeled.")
    else:
        print(f"  WARNINGS — at {int(args.labeled_ratio*100)}% labeled "
              f"(threshold = {args.min_inst_threshold:,} instances/class):")
        for w in warns:
            print(f"    - {w}")

    print()
    print("Done. Inspect the numbers and decide whether to lock the vehicle set.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())