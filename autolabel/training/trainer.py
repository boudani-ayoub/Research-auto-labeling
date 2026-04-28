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
      images/train/{image_filename}
      labels/train/{image_filename_stem}.txt

labeled_data dict format:
  {
    "image_dir":           str  — directory of labeled images
    "label_dir":           str  — directory of YOLO .txt label files
    "image_list":          List[str] — filenames (without path) of labeled images
    "unlabeled_image_dir": str  — directory of unlabeled images (required)
                                  pseudo-label images are looked up here
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
            data    = str(yaml_path),
            epochs  = self.config.epochs,
            batch   = self.config.batch,
            imgsz   = self.config.imgsz,
            device  = self.config.device,
            workers = self.config.workers,
            project = str(round_dir),
            name    = "train",
            exist_ok= True,
            verbose = False,
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
        """
        # Resolve and validate BEFORE touching the filesystem.
        # A validation error must never delete a previously valid assembly.
        labeled = self._resolve_labeled_data(labeled_data)
        self._validate_unique_image_ids(labeled)

        round_dir   = Path(self.config.output_dir) / f"round_{round_id:04d}"
        img_train   = round_dir / "images" / "train"
        label_train = round_dir / "labels" / "train"

        # Fresh directory per round — idempotent assembly.
        # Only reached if validation passed — safe to delete now.
        if round_dir.exists():
            shutil.rmtree(round_dir)
        img_train.mkdir(parents=True, exist_ok=False)
        label_train.mkdir(parents=True, exist_ok=False)

        # ── Copy labeled images and labels ─────────────────────────────────
        for fname in labeled["image_list"]:
            src_img   = Path(labeled["image_dir"]) / fname
            src_label = Path(labeled["label_dir"]) / (Path(fname).stem + ".txt")
            dst_img   = img_train   / fname
            dst_label = label_train / (Path(fname).stem + ".txt")
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)

        # ── Write pseudo-label files ───────────────────────────────────────
        pseudo_by_image: Dict[str, List[PseudoLabel]] = {}
        for pl in pseudo_labels:
            pseudo_by_image.setdefault(pl.image_id, []).append(pl)

        for image_id, pls in pseudo_by_image.items():
            # Locate the source image — raises if not found
            src_img = self._find_image(image_id, labeled)

            fname     = src_img.name
            dst_img   = img_train   / fname
            dst_label = label_train / (src_img.stem + ".txt")

            # Uniqueness already validated at dataset resolution time.
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
        self._write_yaml(round_dir, img_train)
        return round_dir

    def _write_yaml(self, round_dir: Path, img_train: Path) -> None:
        nc    = self.config.num_classes
        names = self.config.class_names or [str(i) for i in range(nc)]
        yaml_data = {
            "path":  str(round_dir),
            "train": str(img_train.relative_to(round_dir)),
            "val":   str(img_train.relative_to(round_dir)),
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
        Enforce MVP uniqueness rule: no image stem may appear in both
        labeled image_dir and unlabeled_image_dir.
        Called immediately after resolving labeled_data, before any I/O.
        Raises ValueError naming the first conflicting stem found.
        """
        labeled_dir   = Path(labeled.get("image_dir", ""))
        unlabeled_dir = Path(labeled.get("unlabeled_image_dir", ""))

        if not labeled_dir.exists() or not unlabeled_dir.exists():
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