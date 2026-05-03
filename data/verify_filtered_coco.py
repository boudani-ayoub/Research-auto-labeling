"""
data/verify_filtered_coco.py
=============================
Verify a filtered-COCO dataset built by make_filtered_coco_splits.py
before any training is launched. Catches silent data corruption that
would otherwise quietly poison experiments.

Checks performed (in order):
  1.  filter_stats.json exists and is valid JSON
  2.  Image symlinks under images/{train2017,val2017}/ resolve to
      readable files in the original COCO source
  3.  Every filtered image has a corresponding .txt label file
      and every .txt has a corresponding image symlink
  4.  Ultralytics /images/->/labels/ path substitution produces
      real, existing paths for a sample of files
  5.  All label files contain only class IDs in {0, 1, 2, 3, 4}
  6.  All YOLO box coordinates are in [0, 1] with positive size
  7.  For the split JSON: train / inner_val / unlabeled are
      pairwise disjoint, and (train + inner_val) == labeled
  8.  For the split JSON: each labeled-train, inner-val, and
      unlabeled image_id has both a symlink image and a label .txt
  9.  Filtered val contains at least one instance of each class
 10.  Each split partition contains at least one image with each
      class (warning, not error, unless --strict)
 11.  Visual overlay: 20 random images saved to visual_overlays/
      with bounding boxes drawn, for human eyeballing

Usage:
    python data/verify_filtered_coco.py \\
        --filtered_root ~/research/coco_filtered_vehicles \\
        --split_file    data/splits_vehicles/coco_vehicles_fold0_10pct.json

    Optional:
        --n_overlays N    (default 20, set 0 to skip)
        --overlay_seed S  (default 12345)
        --strict          (upgrade per-partition class-coverage warnings to errors)

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ── Pretty-print result helpers ──────────────────────────────────────────────

class Result:
    """Accumulator for check results so we can print all outcomes at the end."""
    def __init__(self):
        self.checks: List[Tuple[str, str, str]] = []  # (status, name, detail)

    def ok(self, name: str, detail: str = "") -> None:
        self.checks.append(("OK", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(("WARN", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(("FAIL", name, detail))

    def has_failures(self) -> bool:
        return any(c[0] == "FAIL" for c in self.checks)

    def summary(self, strict: bool) -> int:
        ok    = sum(1 for c in self.checks if c[0] == "OK")
        warn  = sum(1 for c in self.checks if c[0] == "WARN")
        fail  = sum(1 for c in self.checks if c[0] == "FAIL")
        total = len(self.checks)
        print()
        print("=" * 76)
        print(f"  Summary: {ok}/{total} OK, {warn} warn, {fail} fail")
        print("=" * 76)
        for status, name, detail in self.checks:
            tag = {"OK": "  ✓", "WARN": "  !", "FAIL": "  ✗"}[status]
            print(f"{tag} {name}")
            if detail:
                for line in detail.splitlines():
                    print(f"      {line}")
        if fail > 0 or (strict and warn > 0):
            return 1
        return 0


# ── Individual check functions ───────────────────────────────────────────────

def check_filter_stats(filtered_root: Path, r: Result) -> Optional[Dict]:
    """Check 1: filter_stats.json present, parseable, has expected keys."""
    p = filtered_root / "filter_stats.json"
    if not p.exists():
        r.fail("filter_stats.json", f"missing at {p}")
        return None
    try:
        stats = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        r.fail("filter_stats.json", f"invalid JSON: {e}")
        return None

    expected_keys = {"selected_coco_ids", "class_remap", "class_names",
                     "train", "val", "image_dir_train", "image_dir_val",
                     "label_dir_train", "label_dir_val"}
    missing = expected_keys - set(stats.keys())
    if missing:
        r.fail("filter_stats.json keys", f"missing keys: {sorted(missing)}")
        return None

    r.ok("filter_stats.json", f"loaded; classes={stats['class_names']}")
    return stats


def check_symlinks_resolve(image_dir:    Path,
                            expected_n:   int,
                            partition:    str,
                            r:            Result) -> Set[str]:
    """
    Check 2: every symlink under image_dir resolves to a readable file.
    Returns the set of image filenames found.
    """
    if not image_dir.exists():
        r.fail(f"symlink dir [{partition}]", f"missing: {image_dir}")
        return set()

    found_filenames: Set[str] = set()
    broken: List[str] = []
    for entry in sorted(image_dir.iterdir()):
        if not entry.is_symlink() and not entry.is_file():
            continue
        # Resolve. If symlink target is missing, exists() returns False.
        if not entry.exists():
            broken.append(entry.name)
        else:
            found_filenames.add(entry.name)

    if broken:
        sample = broken[:5]
        r.fail(f"symlinks resolve [{partition}]",
               f"{len(broken)} broken symlinks; first 5: {sample}")
        return found_filenames

    if len(found_filenames) != expected_n:
        r.fail(f"symlink count [{partition}]",
               f"expected {expected_n}, found {len(found_filenames)}")
        return found_filenames

    r.ok(f"symlinks resolve [{partition}]",
         f"{len(found_filenames)} symlinks all resolve to source files")
    return found_filenames


def check_label_image_pairing(image_dir:    Path,
                                label_dir:   Path,
                                partition:   str,
                                r:           Result
                                ) -> Tuple[Set[str], Set[str]]:
    """
    Check 3: every image has a label and vice versa.
    Returns (image_stems, label_stems).
    """
    if not label_dir.exists():
        r.fail(f"label dir [{partition}]", f"missing: {label_dir}")
        return set(), set()

    image_stems = {p.stem for p in image_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    label_stems = {p.stem for p in label_dir.iterdir()
                   if p.suffix.lower() == ".txt"}

    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    if images_without_labels:
        sample = sorted(images_without_labels)[:5]
        r.fail(f"label-image pairing [{partition}]",
               f"{len(images_without_labels)} images have no label; "
               f"first 5: {sample}")
        return image_stems, label_stems

    if labels_without_images:
        sample = sorted(labels_without_images)[:5]
        r.fail(f"label-image pairing [{partition}]",
               f"{len(labels_without_images)} labels have no image; "
               f"first 5: {sample}")
        return image_stems, label_stems

    r.ok(f"label-image pairing [{partition}]",
         f"{len(image_stems)} images, all paired with labels")
    return image_stems, label_stems


def check_path_substitution(image_dir:   Path,
                              label_dir:  Path,
                              partition:  str,
                              r:          Result,
                              n_samples:  int = 10) -> None:
    """
    Check 4: simulate Ultralytics' /images/ -> /labels/ path substitution.
    For a sample of images, check the substituted path exists AND points
    inside the filtered tree (not the original COCO labels tree).
    """
    images = sorted(image_dir.iterdir())
    if not images:
        r.fail(f"path substitution [{partition}]", "no images in dir")
        return

    # Sample evenly across the directory
    n = min(n_samples, len(images))
    sample_indices = [i * len(images) // n for i in range(n)]
    samples = [images[i] for i in sample_indices]

    expected_label_root = str(label_dir.parent)  # output_dir/labels
    failures: List[str] = []
    wrong_root: List[str] = []
    for img in samples:
        img_path = str(img)
        label_path = img_path.replace("/images/", "/labels/").replace(
            ".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
        if not Path(label_path).exists():
            failures.append(f"{img.name}: {label_path}")
        elif not label_path.startswith(expected_label_root):
            wrong_root.append(f"{img.name}: lands at {label_path}")

    if failures:
        r.fail(f"path substitution [{partition}]",
               f"{len(failures)}/{n} sampled images don't substitute to "
               f"existing labels; first 3: {failures[:3]}")
        return
    if wrong_root:
        r.fail(f"path substitution [{partition}]",
               f"{len(wrong_root)} substitutions land outside filtered tree "
               f"(should be under {expected_label_root}); first 3: "
               f"{wrong_root[:3]}")
        return

    r.ok(f"path substitution [{partition}]",
         f"{n} sampled image paths substitute correctly to existing filtered labels")


def check_label_content(label_dir:    Path,
                         valid_class_ids: Set[int],
                         partition:    str,
                         r:            Result) -> None:
    """
    Checks 5 + 6: every label line has class_id in valid set, and YOLO
    coordinates in [0,1] with positive size.
    """
    bad_class:   List[Tuple[str, int, str]] = []  # (file, line_no, line)
    bad_coords:  List[Tuple[str, int, str]] = []
    n_lines = 0
    n_files = 0

    for txt in label_dir.iterdir():
        if txt.suffix.lower() != ".txt":
            continue
        n_files += 1
        for i, line in enumerate(txt.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            parts = line.split()
            if len(parts) != 5:
                bad_coords.append((txt.name, i, line))
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:])
            except ValueError:
                bad_coords.append((txt.name, i, line))
                continue
            if cid not in valid_class_ids:
                bad_class.append((txt.name, i, line))
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                bad_coords.append((txt.name, i, line))

    if bad_class:
        sample = bad_class[:3]
        r.fail(f"label class IDs [{partition}]",
               f"{len(bad_class)} lines with class_id outside {sorted(valid_class_ids)}; "
               f"first 3: {sample}")
    else:
        r.ok(f"label class IDs [{partition}]",
             f"{n_lines:,} lines across {n_files:,} files; "
             f"all class IDs in {sorted(valid_class_ids)}")

    if bad_coords:
        sample = bad_coords[:3]
        r.fail(f"label coords [{partition}]",
               f"{len(bad_coords)} malformed lines or out-of-range coords; "
               f"first 3: {sample}")
    else:
        r.ok(f"label coords [{partition}]",
             f"all coordinates in [0,1] with positive w,h")


def check_split_partitions(split:     Dict,
                             r:         Result) -> None:
    """
    Check 7: train/inner_val/unlabeled disjoint, train+inner_val == labeled.
    """
    t = set(split["labeled_train_ids"])
    v = set(split["labeled_inner_val_ids"])
    u = set(split["unlabeled_ids"])
    labeled = set(split["labeled_ids"])

    issues: List[str] = []
    if t & v:
        issues.append(f"train ∩ inner_val: {len(t & v)} overlap")
    if t & u:
        issues.append(f"train ∩ unlabeled: {len(t & u)} overlap")
    if v & u:
        issues.append(f"inner_val ∩ unlabeled: {len(v & u)} overlap")
    if (t | v) != labeled:
        diff_size = len((t | v) ^ labeled)
        issues.append(f"train ∪ inner_val ≠ labeled (|symdiff|={diff_size})")

    if issues:
        r.fail("split partitions disjoint",
               "; ".join(issues))
    else:
        r.ok("split partitions disjoint",
             f"{len(t):,} train + {len(v):,} inner_val + {len(u):,} unlabeled "
             f"(total {len(t)+len(v)+len(u):,} unique image IDs)")


def check_split_ids_have_files(split:        Dict,
                                 image_dir:    Path,
                                 label_dir:    Path,
                                 r:            Result) -> None:
    """
    Check 8: every image_id in train/inner_val/unlabeled partitions has
    both a symlink image and (for labeled) a label file.
    Unlabeled images have symlinks but no label files (they're unlabeled
    by definition).
    """
    all_labeled = set(split["labeled_train_ids"]) | set(split["labeled_inner_val_ids"])
    all_unlabeled = set(split["unlabeled_ids"])

    missing_imgs_labeled:   List[int] = []
    missing_labels_labeled: List[int] = []
    missing_imgs_unlabeled: List[int] = []

    for image_id in all_labeled:
        fname = f"{image_id:012d}"
        if not (image_dir / f"{fname}.jpg").exists():
            missing_imgs_labeled.append(image_id)
        if not (label_dir / f"{fname}.txt").exists():
            missing_labels_labeled.append(image_id)

    for image_id in all_unlabeled:
        if not (image_dir / f"{image_id:012d}.jpg").exists():
            missing_imgs_unlabeled.append(image_id)

    issues: List[str] = []
    if missing_imgs_labeled:
        issues.append(
            f"{len(missing_imgs_labeled)} labeled image_ids missing image symlink "
            f"(first: {missing_imgs_labeled[:3]})"
        )
    if missing_labels_labeled:
        issues.append(
            f"{len(missing_labels_labeled)} labeled image_ids missing .txt label "
            f"(first: {missing_labels_labeled[:3]})"
        )
    if missing_imgs_unlabeled:
        issues.append(
            f"{len(missing_imgs_unlabeled)} unlabeled image_ids missing image symlink "
            f"(first: {missing_imgs_unlabeled[:3]})"
        )

    if issues:
        r.fail("split image_ids have files", "; ".join(issues))
    else:
        r.ok("split image_ids have files",
             f"all {len(all_labeled):,} labeled + {len(all_unlabeled):,} unlabeled "
             f"image_ids have expected symlinks/labels")


def count_class_instances_in_dir(label_dir: Path) -> Dict[int, int]:
    """Helper: scan all .txt files in label_dir and count instances per class_id."""
    counts: Dict[int, int] = defaultdict(int)
    for txt in label_dir.iterdir():
        if txt.suffix.lower() != ".txt":
            continue
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    pass
    return dict(counts)


def check_val_class_presence(label_dir_val: Path,
                               class_names:   List[str],
                               r:             Result) -> None:
    """Check 9: filtered val contains all classes."""
    counts = count_class_instances_in_dir(label_dir_val)
    missing = [class_names[cid] for cid in range(len(class_names))
               if counts.get(cid, 0) == 0]
    if missing:
        r.fail("val class presence",
               f"missing instances for class(es): {missing}")
    else:
        per_class = ", ".join(
            f"{class_names[cid]}={counts.get(cid, 0)}"
            for cid in range(len(class_names))
        )
        r.ok("val class presence", per_class)


def count_class_instances_for_image_ids(label_dir:  Path,
                                          image_ids: Set[int]
                                          ) -> Dict[int, int]:
    """Helper: scan only the specified image_ids' label files."""
    counts: Dict[int, int] = defaultdict(int)
    for image_id in image_ids:
        txt = label_dir / f"{image_id:012d}.txt"
        if not txt.exists():
            continue
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    pass
    return dict(counts)


def check_partition_class_coverage(split:     Dict,
                                     label_dir: Path,
                                     r:         Result) -> None:
    """
    Check 10: each labeled partition (train, inner_val) has at least one
    instance of each class. Warning, not error, by default — at low ratios
    or unfortunate folds, a class may be absent purely from random sampling.
    """
    class_names = split["class_names"]
    n_classes   = len(class_names)

    for partition_name, key in [("labeled_train", "labeled_train_ids"),
                                  ("inner_val",     "labeled_inner_val_ids")]:
        ids = set(split[key])
        counts = count_class_instances_for_image_ids(label_dir, ids)
        missing = [class_names[cid] for cid in range(n_classes)
                   if counts.get(cid, 0) == 0]
        per_class = ", ".join(
            f"{class_names[cid]}={counts.get(cid, 0)}"
            for cid in range(n_classes)
        )
        if missing:
            r.warn(f"class coverage [{partition_name}]",
                   f"missing classes: {missing}\n"
                   f"counts: {per_class}\n"
                   f"(may be acceptable at low ratios; use --strict to fail)")
        else:
            r.ok(f"class coverage [{partition_name}]", per_class)


def visual_overlay(label_dir:   Path,
                    image_dir:  Path,
                    class_names: List[str],
                    out_dir:    Path,
                    n:          int,
                    seed:       int,
                    r:          Result) -> None:
    """
    Check 11: pick n random images, draw bounding boxes from the .txt
    label, save to out_dir for human eyeballing.
    """
    if n <= 0:
        return  # explicitly skipped

    try:
        import cv2  # noqa: F401
    except ImportError:
        r.warn("visual overlays",
               "cv2 (opencv-python) not installed; skipping. "
               "Install with: pip install opencv-python")
        return
    import cv2

    label_files = sorted(label_dir.iterdir())
    label_files = [p for p in label_files if p.suffix.lower() == ".txt"]
    if not label_files:
        r.warn("visual overlays", "no label files found")
        return

    rng = random.Random(seed)
    n_actual = min(n, len(label_files))
    chosen = rng.sample(label_files, n_actual)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 5 distinct colours (BGR for OpenCV)
    colours = [
        (255, 0, 0),    # bicycle — blue
        (0, 255, 0),    # car — green
        (0, 0, 255),    # motorcycle — red
        (255, 255, 0),  # bus — cyan
        (255, 0, 255),  # truck — magenta
    ]

    n_saved = 0
    failures: List[str] = []
    for txt in chosen:
        img_path = image_dir / f"{txt.stem}.jpg"
        if not img_path.exists():
            failures.append(f"{txt.stem}: image symlink missing")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            failures.append(f"{txt.stem}: cv2.imread failed")
            continue
        h, w = img.shape[:2]

        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:])
            except ValueError:
                continue
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            colour = colours[cid % len(colours)]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            label = class_names[cid] if 0 <= cid < len(class_names) else f"?{cid}"
            cv2.putText(img, label, (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        out_path = out_dir / f"{txt.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        n_saved += 1

    detail = f"saved {n_saved} overlays to {out_dir}"
    if failures:
        detail += f"\n  {len(failures)} failures: {failures[:3]}"
    r.ok("visual overlays", detail)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a filtered-COCO dataset before training"
    )
    parser.add_argument("--filtered_root", required=True, type=Path,
        help="Output root from make_filtered_coco_splits.py "
             "(e.g. ~/research/coco_filtered_vehicles)")
    parser.add_argument("--split_file", required=True, type=Path,
        help="Path to one split JSON to verify "
             "(e.g. data/splits_vehicles/coco_vehicles_fold0_10pct.json)")
    parser.add_argument("--n_overlays", type=int, default=20,
        help="Number of visual overlay images to save (default 20, "
             "set 0 to skip)")
    parser.add_argument("--overlay_seed", type=int, default=12345,
        help="Seed for overlay image sampling (default 12345)")
    parser.add_argument("--strict", action="store_true",
        help="Treat partition-class-coverage warnings as failures")
    args = parser.parse_args()

    filtered_root = args.filtered_root.expanduser().resolve()
    split_file    = args.split_file.expanduser().resolve()
    r             = Result()

    print()
    print("=" * 76)
    print(f"  Verifying filtered COCO at: {filtered_root}")
    print(f"  Against split:              {split_file}")
    print("=" * 76)
    print()

    # ── Check 1: filter_stats.json ───────────────────────────────────────────
    stats = check_filter_stats(filtered_root, r)
    if stats is None:
        return r.summary(args.strict)

    image_dir_train = Path(stats["image_dir_train"])
    image_dir_val   = Path(stats["image_dir_val"])
    label_dir_train = Path(stats["label_dir_train"])
    label_dir_val   = Path(stats["label_dir_val"])
    class_names     = stats["class_names"]
    n_classes       = len(class_names)
    valid_class_ids = set(range(n_classes))
    expected_train  = stats["train"]["filtered_image_ids"]
    expected_val    = stats["val"]["filtered_image_ids"]

    # ── Check 2: symlinks resolve ────────────────────────────────────────────
    train_files = check_symlinks_resolve(
        image_dir_train, expected_train, "train", r
    )
    val_files = check_symlinks_resolve(
        image_dir_val, expected_val, "val", r
    )

    # ── Check 3: image-label pairing ─────────────────────────────────────────
    check_label_image_pairing(image_dir_train, label_dir_train, "train", r)
    check_label_image_pairing(image_dir_val,   label_dir_val,   "val",   r)

    # ── Check 4: path substitution ───────────────────────────────────────────
    check_path_substitution(image_dir_train, label_dir_train, "train", r)
    check_path_substitution(image_dir_val,   label_dir_val,   "val",   r)

    # ── Checks 5 + 6: label content ──────────────────────────────────────────
    check_label_content(label_dir_train, valid_class_ids, "train", r)
    check_label_content(label_dir_val,   valid_class_ids, "val",   r)

    # ── Load split for partition checks ──────────────────────────────────────
    if not split_file.exists():
        r.fail("split file", f"missing: {split_file}")
        return r.summary(args.strict)
    try:
        split = json.loads(split_file.read_text())
    except json.JSONDecodeError as e:
        r.fail("split file", f"invalid JSON: {e}")
        return r.summary(args.strict)

    # ── Check 7: partition disjointness ──────────────────────────────────────
    check_split_partitions(split, r)

    # ── Check 8: split image_ids have files ──────────────────────────────────
    check_split_ids_have_files(split, image_dir_train, label_dir_train, r)

    # ── Check 9: val class presence ──────────────────────────────────────────
    check_val_class_presence(label_dir_val, class_names, r)

    # ── Check 10: per-partition class coverage ───────────────────────────────
    check_partition_class_coverage(split, label_dir_train, r)

    # ── Check 11: visual overlays ────────────────────────────────────────────
    overlay_dir = filtered_root / "visual_overlays"
    visual_overlay(
        label_dir   = label_dir_train,
        image_dir   = image_dir_train,
        class_names = class_names,
        out_dir     = overlay_dir,
        n           = args.n_overlays,
        seed        = args.overlay_seed,
        r           = r,
    )

    return r.summary(args.strict)


if __name__ == "__main__":
    sys.exit(main())