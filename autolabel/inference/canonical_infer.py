"""
inference/canonical_infer.py
============================
Owns canonical inference. Runs YOLOv8 on images with fixed settings,
returns a CandidatePool (Dict[image_id, List[PseudoLabel]]).

Architecture constraints (frozen):
  - Uses ClassScoreCapturingPredictor from Phase 0 spike (preferred path)
  - Fixed imgsz, conf threshold, NMS iou — identical across all rounds
  - No augmentation: no Mosaic, no MixUp, no flips, no TTA
  - PseudoLabel.confidence   ← results.boxes.conf     (admission gate ONLY)
  - PseudoLabel.class_scores ← softmax(logits)[anchor] (scoring / JS ONLY)
  - These two fields are populated independently — no equivalence assumed
  - box_id convention: f"{image_id}_r{round_id}_{det_index:04d}"
  - May cache predictions by (checkpoint_hash, image_id) — optional
  - GPU inference, CPU post-processing of class scores

Phase 0 result locked in:
  - torch 2.5.1+cu121, ultralytics 8.4.38, RTX 4090
  - ClassScoreCapturingPredictor with letterbox coordinate fix (Method B)
  - All 4 spike checks passed on 80 real detections
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor

from autolabel.bank.schemas import CandidatePool, PseudoLabel


# ── ClassScoreCapturingPredictor (Phase 0 preferred path) ────────────────────

class ClassScoreCapturingPredictor(DetectionPredictor):
    """
    Subclasses DetectionPredictor and overrides postprocess() to capture
    the full K-dimensional class probability vector per detection before
    NMS discards non-kept anchors.

    Phase 0 verified:
      CHECK 1: shape [B, num_classes, num_anchors] confirmed
      CHECK 2: argmax(class_scores) == results.boxes.cls for all detections
      CHECK 3: confidence and class_scores populated independently
      CHECK 4: mapping is bit-identical across repeated calls

    Method B (letterbox coordinate fix):
      Kept boxes from results are in ORIGINAL IMAGE space.
      Anchor boxes are in MODEL INPUT space (imgsz × imgsz).
      We transform kept boxes → model input space using the letterbox formula
      before matching, giving exact coordinate identity.
    """

    def postprocess(self, preds, img, orig_imgs):
        raw = preds[0]                                          # [B, 4+nc, N]
        B, channels, N = raw.shape
        nc = channels - 4

        self._raw_logits_shape     = (B, nc, N)
        self._captured_class_probs = torch.softmax(
            raw[:, 4:, :], dim=1).detach().cpu()               # [B, nc, N]
        self._anchor_coords        = raw[:, :4, :].detach().cpu()  # [B, 4, N]
        self._imgsz                = img.shape[-1]              # model input size

        results = super().postprocess(preds, img, orig_imgs)
        self._attach_class_scores(results)
        return results

    def _attach_class_scores(self, results) -> None:
        for b, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                result._class_scores = torch.zeros(
                    0, self._raw_logits_shape[1])
                continue

            nc    = self._captured_class_probs.shape[1]
            imgsz = self._imgsz

            # ── Method B: letterbox-corrected coordinate matching ────────────
            # Phase 0 result: Method B passed all 4 checks on 80 real
            # detections. Method A (col-6 anchor index) is not used —
            # col 6 may contain tracking metadata in some Ultralytics
            # builds and would silently poison class scores if triggered.
            orig_h, orig_w = result.orig_shape
            ratio  = min(imgsz / orig_h, imgsz / orig_w)
            pad_w  = (imgsz - orig_w * ratio) / 2.0
            pad_h  = (imgsz - orig_h * ratio) / 2.0

            kept_xyxy = boxes.data[:, :4].cpu()                 # orig image space

            # Transform kept boxes → model input space (letterbox inverse)
            kept_model          = kept_xyxy.clone()
            kept_model[:, 0]    = kept_xyxy[:, 0] * ratio + pad_w
            kept_model[:, 1]    = kept_xyxy[:, 1] * ratio + pad_h
            kept_model[:, 2]    = kept_xyxy[:, 2] * ratio + pad_w
            kept_model[:, 3]    = kept_xyxy[:, 3] * ratio + pad_h

            # Decode anchor xywh → xyxy in model input space
            ab   = self._anchor_coords[b]                       # [4, N]
            ax1  = ab[0] - ab[2] / 2
            ay1  = ab[1] - ab[3] / 2
            ax2  = ab[0] + ab[2] / 2
            ay2  = ab[1] + ab[3] / 2
            anchor_xyxy = torch.stack([ax1, ay1, ax2, ay2], dim=1)  # [N, 4]

            dists       = torch.cdist(
                kept_model.float(), anchor_xyxy.float(), p=2)
            best_anchor = dists.argmin(dim=1)
            scores_out  = self._captured_class_probs[
                b, :, best_anchor].T                            # [n_kept, nc]

            result._class_scores = scores_out


# ── Canonical inference entry point ──────────────────────────────────────────

def canonical_infer(model:      YOLO,
                    image_ids:  List[str],
                    images:     List[np.ndarray],
                    round_id:   int,
                    conf:       float = 0.05,
                    iou_nms:    float = 0.45,
                    imgsz:      int   = 640,
                    batch_size: int   = 8) -> CandidatePool:
    """
    Run canonical inference on a list of images and return a CandidatePool.

    Batch composition contract (frozen):
      class_scores depend on letterbox padding which depends on batch
      composition. To prevent cross-round instability in C_cls_dist:
        - image_ids are SORTED lexicographically before batching (enforced)
        - batch_size must be identical across all rounds and resume runs
        - imgsz must be identical across all rounds
        - Do NOT mix batch and single-image inference for C_t generation

    Args:
        model:      loaded YOLO model (ultralytics.YOLO)
        image_ids:  list of image identifiers, one per image
        images:     list of uint8 BGR numpy arrays [H, W, C]
        round_id:   current pipeline round (used for box_id generation)
        conf:       minimum confidence threshold (intentionally loose, e.g. 0.05)
        iou_nms:    NMS IoU threshold
        imgsz:      model input size — must be identical across all rounds
        batch_size: images per inference batch — must be identical across all
                    rounds and resume runs (part of experiment config)

    Returns:
        CandidatePool: Dict[image_id, List[PseudoLabel]]
        Images with zero detections are included with an empty list.
        Pool is keyed by image_id regardless of input order.

    PseudoLabel fields:
        confidence   ← results.boxes.conf        (admission gate ONLY)
        class_scores ← softmax(logits)[anchor]   (scoring / JS ONLY)
    """
    if len(image_ids) != len(images):
        raise ValueError(
            f"canonical_infer: image_ids length ({len(image_ids)}) must "
            f"match images length ({len(images)})"
        )

    # ── Batch composition contract: sort by image_id ──────────────────────────
    # Fixes batch composition across rounds so letterbox padding is identical
    # for the same image in every round, making class_scores stable.
    sorted_pairs = sorted(zip(image_ids, images), key=lambda x: x[0])
    sorted_ids   = [p[0] for p in sorted_pairs]
    sorted_imgs  = [p[1] for p in sorted_pairs]

    # GPU determinism — required for consistent class_scores across calls
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    pool: CandidatePool = {}

    # Process in fixed-size chunks to lock batch composition per chunk
    for chunk_start in range(0, len(sorted_ids), batch_size):
        chunk_ids  = sorted_ids[chunk_start : chunk_start + batch_size]
        chunk_imgs = sorted_imgs[chunk_start : chunk_start + batch_size]

        predictor = ClassScoreCapturingPredictor(
            overrides={
                "model":   model.model,
                "conf":    conf,
                "iou":     iou_nms,
                "imgsz":   imgsz,
                "verbose": False,
            }
        )
        predictor.setup_model(model=model.model)
        results = predictor(source=chunk_imgs, stream=False)

        for image_id, result in zip(chunk_ids, results):
            pls: List[PseudoLabel] = []

            if result.boxes is not None and len(result.boxes) > 0:
                nc          = predictor._raw_logits_shape[1]
                cs_tensor   = result._class_scores          # [n_kept, nc]
                confidences = result.boxes.conf.cpu()
                classes     = result.boxes.cls.cpu().long()
                xyxy        = result.boxes.xyxy.cpu()

                for det_idx in range(len(result.boxes)):
                    cs     = cs_tensor[det_idx]             # [nc]
                    cs_sum = float(cs.sum())
                    if cs_sum > 1e-8:
                        cs = cs / cs_sum
                    else:
                        cs = torch.ones(nc, dtype=torch.float32) / nc

                    conf_val = float(confidences[det_idx])
                    cls_val  = int(classes[det_idx])
                    box_xyxy = tuple(float(v) for v in xyxy[det_idx])

                    pl = PseudoLabel(
                        image_id    = image_id,
                        box_id      = f"{image_id}_r{round_id}_{det_idx:04d}",
                        round_id    = round_id,
                        box         = box_xyxy,
                        pred_class  = cls_val,
                        class_scores= tuple(float(v) for v in cs),
                        confidence  = conf_val,
                    )
                    pls.append(pl)

            pool[image_id] = pls

    return pool


# ── Checkpoint hash helper (for optional caching) ────────────────────────────

def checkpoint_hash(checkpoint_path: str) -> str:
    """
    Return a short SHA256 hash of a checkpoint file.
    Used as the cache key component for (checkpoint_hash, image_id) caching.
    Only hashes the first 1MB for speed — sufficient to distinguish rounds.
    """
    h   = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        chunk = f.read(1024 * 1024)
        h.update(chunk)
    return h.hexdigest()[:16]