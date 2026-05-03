"""
autolabel/training/trainer.py
==============================
Owns YOLOv8 training interface.

Architecture constraints (frozen):
  - ONLY module that imports or touches ultralytics training code
  - Assembles labeled + pseudo dataset on disk in YOLO format
  - Writes dataset.yaml
  - Calls ultralytics.YOLO.train()
  - Returns checkpoint path (str)

YOLO label format (one file per image, normalised to [0,1]):
  <class_id> <cx> <cy> <w> <h>

Dataset directory layout:
  {output_dir}/round_{round_id:04d}/
      dataset.yaml
      images/train/{image_filename}      # labeled 90% + admitted pseudo-label images
      labels/train/{image_filename_stem}.txt
      images/val/{image_filename}        # labeled inner-val 10% (frozen, optional)
      labels/val/{image_filename_stem}.txt

Inner val (Option A asymmetric protocol):
  When labeled_data carries a non-empty `inner_val_image_list`, the trainer
  assembles a SEPARATE inner-val partition under images/val/ + labels/val/
  using ground-truth labels from `label_dir`. dataset.yaml then points
  `val:` at images/val/.

  Pseudo-labels NEVER enter images/val/ or labels/val/. The inner-val set
  is frozen across all rounds and is consumed only by YOLO's per-run
  patience-based early stopping. The outer loop (admission, stopping)
  remains validation-free.

  When `inner_val_image_list` is absent or empty (legacy / unit-test path),
  the trainer falls back to `val: images/train/` for backward compatibility.

labeled_data dict format:
  {
    "image_dir":            str       — directory of labeled images
    "label_dir":            str       — directory of YOLO .txt label files
    "image_list":           List[str] — filenames of labeled training images (90%)
    "inner_val_image_list": List[str] — filenames of inner-val images (10%, optional)
    "unlabeled_image_dir":  str       — directory of unlabeled images
    "unlabeled_list":       List[str] — filenames of unlabeled images (optional,
                                        used in same-dir disjointness check)
  }

Checkpoint protocol (explicit):
  train() accepts an optional init_checkpoint argument.
  Option A (continue from prev checkpoint): pass init_checkpoint=<path>
  Option B (fixed base model each round):   pass init_checkpoint=None
  The orchestrator decides which to use — the trainer does not choose.
  For MVP, Option B is the default (pretrained YOLOv8m each round).
"""

from __future__ import annotations

import os
import shutil
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from autolabel.bank.schemas import PseudoLabel
from autolabel.orchestrator import TrainerInterface


@dataclass
class TrainingConfig:
    """
    Configuration for YOLOv8 training.
    Matches the training section of configs/default.yaml.

    patience: epochs of no inner-val improvement before early stopping fires.
              Only meaningful when an inner-val list is supplied. Default 100
              matches Ultralytics' default (effectively disables early stopping
              for epochs <= 100). Lower (e.g. 50) to leverage Option A inner val.

    optimizer: explicit optimizer choice. Default "AdamW".
               Setting this avoids Ultralytics' `optimizer=auto` picking
               different optimizers depending on dataset size and epoch
               count — which is a reproducibility hazard. Observed in the
               wild: auto picks AdamW at 2 epochs but MuSGD/SGD at 100,
               and MuSGD with lr=0.01 catastrophically destroys pretrained
               weights at the start of training.

    lr0: initial learning rate. Default 0.001 — the standard AdamW
         transfer-learning rate for object detection. Keep paired with
         the optimizer choice — lr0=0.01 is appropriate for SGD but
         destroys AdamW finetuning, and vice versa.
    """
    epochs:      int   = 100
    batch:       int   = 16
    device:      str   = "0"
    workers:     int   = 8
    pretrained:  bool  = True
    base_model:  str   = "yolov8m.pt"
    imgsz:       int   = 640
    output_dir:  str   = "outputs"
    num_classes: int   = 80
    class_names: Optional[List[str]] = None
    patience:    int   = 100
    optimizer:   str   = "AdamW"
    lr0:         float = 0.001


class YOLOTrainer(TrainerInterface):
    """
    Assembles a YOLO-format dataset from labeled data + admitted pseudo-labels,
    writes dataset.yaml, and calls ultralytics.YOLO.train().

    This is the only module that imports ultralytics training code.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self,
              labeled_data:    Union[Dict, str],
              pseudo_labels:   List[PseudoLabel],
              round_id:        int,
              init_checkpoint: Optional[str] = None) -> str:
        """
        Assemble dataset, write yaml, train, return checkpoint path.

        Args:
            labeled_data:    labeled_data dict (must include unlabeled_image_dir)
            pseudo_labels:   A_t — admitted pseudo-labels (empty for round 0)
            round_id:        current round
            init_checkpoint: Optional path to a .pt file to start training from.
                             None → start from self.config.base_model (Option B).
                             <path> → start from that checkpoint (Option A).
                             The orchestrator decides which option to use.

        Returns:
            str: path to best.pt checkpoint from this training run
        """
        round_dir = self._prepare_dataset(labeled_data, pseudo_labels, round_id)
        yaml_path = round_dir / "dataset.yaml"

        from ultralytics import YOLO
        start_weights = init_checkpoint if init_checkpoint else self.config.base_model
        model = YOLO(start_weights)

        model.train(
            data      = str(yaml_path),
            epochs    = self.config.epochs,
            batch     = self.config.batch,
            imgsz     = self.config.imgsz,
            device    = self.config.device,
            workers   = self.config.workers,
            patience  = self.config.patience,
            optimizer = self.config.optimizer,
            lr0       = self.config.lr0,
            project   = str(round_dir),
            name      = "train",
            exist_ok  = True,
            verbose   = False,
        )

        checkpoint = round_dir / "train" / "weights" / "best.pt"
        if not checkpoint.exists():
            candidates = list(round_dir.rglob("best.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"Training completed but best.pt not found under {round_dir}"
                )
            checkpoint = candidates[0]

        return str(checkpoint)

    # ── Dataset assembly ──────────────────────────────────────────────────────

    def _prepare_dataset(self,
                          labeled_data:  Union[Dict, str],
                          pseudo_labels: List[PseudoLabel],
                          round_id:      int) -> Path:
        """
        Build the YOLO dataset directory for this round.
        Returns the round directory path.
        Raises FileNotFoundError if a pseudo-label image cannot be located.

        Inner-val partition (Option A protocol):
          When labeled_data["inner_val_image_list"] is non-empty, a
          separate images/val/ + labels/val/ pair is written from
          labeled image_dir + label_dir (ground truth, not pseudo).
          Pseudo-labels are NEVER written into the val partition.
          Frozen across rounds for stable per-run early stopping.
        """
        # Resolve and validate ALL inputs BEFORE touching the filesystem.
        # A validation error must never delete a previously valid assembly.
        labeled = self._resolve_labeled_data(labeled_data)
        self._validate_unique_image_ids(labeled)
        self._validate_pseudo_labels_disjoint_from_inner_val(labeled, pseudo_labels)

        # label_dir check must also happen before rmtree
        label_dir_path = Path(labeled.get("label_dir", ""))
        if label_dir_path and not label_dir_path.exists():
            raise FileNotFoundError(
                f"label_dir not found: {label_dir_path}\n"
                f"YOLO-format label files must exist before training. "
                f"For COCO, run: python data/make_splits.py --convert_labels "
                f"--label_output_dir <path> first."
            )

        # Decide whether to write a separate inner-val partition.
        inner_val_image_list = labeled.get("inner_val_image_list") or []
        use_inner_val = len(inner_val_image_list) > 0

        # Resolve to absolute path — Ultralytics prepends runs/detect/ to
        # relative project paths, causing checkpoint lookup to fail.
        round_dir   = Path(self.config.output_dir).resolve() / f"round_{round_id:04d}"
        img_train   = round_dir / "images" / "train"
        label_train = round_dir / "labels" / "train"
        img_val     = round_dir / "images" / "val"
        label_val   = round_dir / "labels" / "val"

        # Fresh directory per round — idempotent assembly.
        # Only reached after all validation passes — safe to delete now.
        if round_dir.exists():
            shutil.rmtree(round_dir)
        img_train.mkdir(parents=True, exist_ok=False)
        label_train.mkdir(parents=True, exist_ok=False)
        if use_inner_val:
            img_val.mkdir(parents=True, exist_ok=False)
            label_val.mkdir(parents=True, exist_ok=False)

        # ── Copy labeled training images and labels (90%) ──────────────────
        self._copy_image_label_pairs(
            filenames     = labeled["image_list"],
            src_image_dir = Path(labeled["image_dir"]),
            src_label_dir = Path(labeled["label_dir"]),
            dst_image_dir = img_train,
            dst_label_dir = label_train,
        )

        # ── Copy inner-val images and ground-truth labels (10%) ────────────
        # Frozen across rounds. No pseudo-labels here — strictly ground truth.
        if use_inner_val:
            self._copy_image_label_pairs(
                filenames     = inner_val_image_list,
                src_image_dir = Path(labeled["image_dir"]),
                src_label_dir = Path(labeled["label_dir"]),
                dst_image_dir = img_val,
                dst_label_dir = label_val,
            )

        # ── Write pseudo-label files (TRAIN ONLY, never val) ───────────────
        pseudo_by_image: Dict[str, List[PseudoLabel]] = {}
        for pl in pseudo_labels:
            pseudo_by_image.setdefault(pl.image_id, []).append(pl)

        for image_id, pls in pseudo_by_image.items():
            # Locate the source image — raises if not found
            src_img = self._find_image(image_id, labeled)

            fname     = src_img.name
            dst_img   = img_train   / fname
            dst_label = label_train / (src_img.stem + ".txt")

            # Disjointness already validated.
            # Safe to copy without collision checks here.
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            h, w = self._get_image_dims(src_img)
            lines = []
            for pl in pls:
                line = self._pl_to_yolo_line(pl, h, w)
                if line is not None:
                    lines.append(line)

            if lines:
                mode = "a" if dst_label.exists() else "w"
                with open(dst_label, mode) as f:
                    f.write("\n".join(lines) + "\n")

        # ── Write dataset.yaml ─────────────────────────────────────────────
        self._write_yaml(
            round_dir = round_dir,
            img_train = img_train,
            img_val   = img_val if use_inner_val else None,
        )
        return round_dir

    def _copy_image_label_pairs(self,
                                 filenames:     List[str],
                                 src_image_dir: Path,
                                 src_label_dir: Path,
                                 dst_image_dir: Path,
                                 dst_label_dir: Path) -> None:
        """
        Copy each (image, label) pair from src to dst.

        Missing source files are skipped silently to match the prior
        behavior of `_prepare_dataset` for the labeled-train pass — the
        upstream label-conversion step is responsible for label
        completeness, not the trainer.
        """
        for fname in filenames:
            src_img   = src_image_dir / fname
            src_label = src_label_dir / (Path(fname).stem + ".txt")
            dst_img   = dst_image_dir / fname
            dst_label = dst_label_dir / (Path(fname).stem + ".txt")
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)

    def _write_yaml(self,
                     round_dir: Path,
                     img_train: Path,
                     img_val:   Optional[Path] = None) -> None:
        """
        Write dataset.yaml. If `img_val` is provided, point `val:` there;
        otherwise fall back to `val: <train>` (legacy / unit-test path).
        """
        nc    = self.config.num_classes
        names = self.config.class_names or [str(i) for i in range(nc)]
        val_dir = img_val if img_val is not None else img_train
        yaml_data = {
            "path":  str(round_dir),
            "train": str(img_train.relative_to(round_dir)),
            "val":   str(val_dir.relative_to(round_dir)),
            "nc":    nc,
            "names": names,
        }
        with open(round_dir / "dataset.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

    def _resolve_labeled_data(self, labeled_data: Union[Dict, str]) -> Dict:
        """Normalise labeled_data to a dict."""
        if isinstance(labeled_data, dict):
            return labeled_data
        base = Path(labeled_data)
        img_dir   = base / "images" / "train"
        label_dir = base / "labels" / "train"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Expected images/train/ under {base}, not found.")
        image_list = [f.name for f in sorted(img_dir.iterdir())
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        return {
            "image_dir":           str(img_dir),
            "label_dir":           str(label_dir),
            "image_list":          image_list,
            "unlabeled_image_dir": str(base / "unlabeled" / "images"),
        }

    def _validate_unique_image_ids(self, labeled: Dict) -> None:
        """
        Enforce image ID uniqueness between labeled, unlabeled, and inner-val sets.

        Two modes for labeled vs unlabeled disjointness:
          SAME directory (e.g. COCO train2017):
            Both labeled image_list and unlabeled_list come from the same dir.
            Validate list-level disjointness: no stem in image_list may also
            appear in unlabeled_list.

          DIFFERENT directories (e.g. separate labeled/unlabeled datasets):
            Validate directory-level disjointness: no stem in labeled image_dir
            may appear in unlabeled_image_dir.

        Inner-val invariants (when inner_val_image_list is non-empty):
          inner_val_image_list ∩ image_list      = ∅   (sub-split is disjoint)
          inner_val_image_list ∩ unlabeled_list  = ∅   (inner val never aliases unlabeled)

        Raises ValueError naming the first conflicting stem found.
        """
        labeled_dir   = Path(labeled.get("image_dir", ""))
        unlabeled_dir = Path(labeled.get("unlabeled_image_dir", ""))

        # ── Inner-val invariants (cheap list-level checks, always run) ─────
        inner_val_image_list = labeled.get("inner_val_image_list") or []
        if inner_val_image_list:
            image_list     = labeled.get("image_list", [])
            unlabeled_list = labeled.get("unlabeled_list", [])
            train_stems     = {Path(f).stem for f in image_list}
            inner_val_stems = {Path(f).stem for f in inner_val_image_list}
            unlabeled_stems = {Path(f).stem for f in unlabeled_list}

            overlap_iv_train = inner_val_stems & train_stems
            if overlap_iv_train:
                example = next(iter(sorted(overlap_iv_train)))
                raise ValueError(
                    f"Inner-val/train overlap: stem {example!r} appears in both "
                    f"image_list (train) and inner_val_image_list. These must be "
                    f"disjoint — split bug in make_splits.py.split_to_labeled_data."
                )
            if unlabeled_list:
                overlap_iv_unl = inner_val_stems & unlabeled_stems
                if overlap_iv_unl:
                    example = next(iter(sorted(overlap_iv_unl)))
                    raise ValueError(
                        f"Inner-val/unlabeled overlap: stem {example!r} appears "
                        f"in both inner_val_image_list and unlabeled_list. "
                        f"Inner val must never alias an unlabeled image — split bug."
                    )

        # ── Existing labeled vs unlabeled disjointness ─────────────────────
        if not unlabeled_dir or not unlabeled_dir.exists():
            return  # no unlabeled dir — nothing further to check

        same_dir = (labeled_dir.resolve() == unlabeled_dir.resolve())

        if same_dir:
            # COCO-style: both sets live in the same directory.
            # Validate list-level disjointness only.
            labeled_list   = labeled.get("image_list", [])
            unlabeled_list = labeled.get("unlabeled_list", [])
            if not unlabeled_list:
                return  # no unlabeled list provided — skip

            labeled_stems   = {Path(f).stem for f in labeled_list}
            unlabeled_stems = {Path(f).stem for f in unlabeled_list}
            overlap = labeled_stems & unlabeled_stems
            if overlap:
                example = next(iter(sorted(overlap)))
                raise ValueError(
                    f"Image ID collision in shared directory: stem {example!r} "
                    f"appears in both image_list and unlabeled_list. "
                    f"Labeled and unlabeled image IDs must be disjoint."
                )
        else:
            # Different directories: directory-level scan
            if not labeled_dir.exists():
                return
            labeled_stems = {
                p.stem for p in labeled_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            }
            for p in unlabeled_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    if p.stem in labeled_stems:
                        raise ValueError(
                            f"Image ID collision: stem {p.stem!r} exists in both "
                            f"labeled image_dir ({labeled_dir}) and "
                            f"unlabeled_image_dir ({unlabeled_dir}). "
                            f"Labeled and unlabeled image IDs must be globally unique."
                        )

    def _validate_pseudo_labels_disjoint_from_inner_val(
            self,
            labeled:       Dict,
            pseudo_labels: List[PseudoLabel]) -> None:
        """
        Pseudo-labels must NEVER target an inner-val image.

        Inner-val images are frozen ground truth used by YOLO for per-run
        early stopping. A pseudo-label landing on an inner-val image would:
          (a) overwrite or append to the ground-truth label file in val/
              (we route pseudo-labels to train/ — so the image would just
              not exist in val/ — but the cleaner failure is to refuse here);
          (b) make the inner val no longer a clean ground-truth signal;
          (c) silently bias YOLO's early-stopping decision.

        Pseudo-labels come from C_t (predictions on unlabeled images).
        labeled_inner_val_ids ⊂ labeled_ids, and labeled_ids ∩ unlabeled_ids = ∅,
        so this check should NEVER fire in a correctly assembled pipeline.
        Fire defensively to surface upstream bugs.
        """
        inner_val_image_list = labeled.get("inner_val_image_list") or []
        if not inner_val_image_list or not pseudo_labels:
            return
        inner_val_stems = {Path(f).stem for f in inner_val_image_list}
        for pl in pseudo_labels:
            if pl.image_id in inner_val_stems:
                raise ValueError(
                    f"Pseudo-label references inner-val image_id "
                    f"{pl.image_id!r}. Inner val must remain frozen ground "
                    f"truth — pseudo-labels never touch the val partition. "
                    f"This indicates an upstream bug: a pseudo-label was "
                    f"generated on a labeled-set image, which should be "
                    f"impossible if C_t is restricted to unlabeled images."
                )

    def _find_image(self, image_id: str, labeled: Dict) -> Path:
        """
        Find the image file for a given image_id.
        Searches labeled image_dir first, then unlabeled_image_dir.
        Raises FileNotFoundError if not found in either location.
        Silent skipping is not acceptable — missing images are pipeline bugs.
        """
        search_dirs = [labeled.get("image_dir", "")]
        unlabeled_dir = labeled.get("unlabeled_image_dir", "")
        if unlabeled_dir:
            search_dirs.append(unlabeled_dir)

        for search_dir in search_dirs:
            d = Path(search_dir)
            if not d.exists():
                continue
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = d / (image_id + ext)
                if candidate.exists():
                    return candidate
            for f in d.iterdir():
                if f.stem == image_id or f.name == image_id:
                    return f

        raise FileNotFoundError(
            f"Image for pseudo-label image_id={image_id!r} not found in "
            f"labeled image_dir or unlabeled_image_dir. "
            f"Searched: {search_dirs}. "
            f"This is a pipeline error — every pseudo-label must reference "
            f"an accessible image."
        )

    def _get_image_dims(self, img_path: Path) -> Tuple[int, int]:
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                return img.shape[:2]
        except Exception:
            pass
        return 640, 640

    def _pl_to_yolo_line(self,
                          pl:    PseudoLabel,
                          img_h: int,
                          img_w: int) -> Optional[str]:
        """Convert PseudoLabel to normalised YOLO line. Returns None if degenerate."""
        x1, y1, x2, y2 = pl.box
        # Clip to image bounds before normalising — defensive against
        # boxes that drifted outside frame during jitter or rounding.
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        nw = bw / img_w
        nh = bh / img_h
        if nw <= 0 or nh <= 0:
            return None
        return f"{pl.pred_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


# ── Dataset assembly helper (dry-run, no training) ────────────────────────────

def assemble_dataset_only(labeled_data:  Union[Dict, str],
                           pseudo_labels: List[PseudoLabel],
                           round_id:      int,
                           config:        TrainingConfig) -> Path:
    """
    Assemble a YOLO dataset on disk without running training.
    Returns the round directory path.
    """
    trainer = YOLOTrainer(config)
    return trainer._prepare_dataset(labeled_data, pseudo_labels, round_id)