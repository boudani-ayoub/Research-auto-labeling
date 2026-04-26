"""
inference/jitter.py
===================
Owns the three deterministic weak image-level perturbations used for
jitter-localization stability scoring.

Architecture constraints (frozen):
  - Exactly 3 transforms: scale (±3%), translation (±8px), brightness (±10%)
  - Applied ONE AT A TIME, never composed
  - Returns (jittered_image, remapped_boxes) — nothing else
  - Does NOT run YOLO inference
  - Does NOT compute IoU
  - Does NOT select detections
  - Parameters are hard-coded constants — not config-driven
  - Orchestrator owns jitter inference end-to-end

Transform directions:
  - scale:      positive delta (+3%) — image scaled up around center
  - translate:  positive dx and dy (+8px) — image shifted right and down
  - brightness: positive delta (+10%) — image brightened

Box remapping is delegated to utils/box_transform.py for the geometric
transforms. Brightness has no geometric effect on boxes.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from autolabel.utils.box_transform import (
    BRIGHTNESS_DELTA,
    SCALE_DELTA,
    TRANSLATE_DELTA,
    remap_box_brightness,
    remap_box_scale,
    remap_box_translate,
)
from autolabel.bank.schemas import PseudoLabel

# Hard-coded jitter parameters — must match utils/box_transform.py constants
_SCALE_FACTOR       = 1.0 + SCALE_DELTA     # 1.03
_TRANSLATE_DX       = float(TRANSLATE_DELTA) # 8.0 px
_TRANSLATE_DY       = float(TRANSLATE_DELTA) # 8.0 px
_BRIGHTNESS_FACTOR  = 1.0 + BRIGHTNESS_DELTA # 1.10

Box = Tuple[float, float, float, float]      # (x1, y1, x2, y2)


# ── Public API ────────────────────────────────────────────────────────────────

def apply_scale(image: np.ndarray,
                boxes: List[Box]) -> Tuple[np.ndarray, List[Box]]:
    """
    Scale the image by +3% around its center.
    Remaps all boxes by the same transform.

    Returns:
        (jittered_image, remapped_boxes)
        Image is the same spatial size as input (resized back after scaling).
    """
    h, w = image.shape[:2]
    jittered = _scale_image(image, _SCALE_FACTOR)
    remapped = [remap_box_scale(b, h, w, _SCALE_FACTOR) for b in boxes]
    return jittered, remapped


def apply_translate(image: np.ndarray,
                    boxes: List[Box]) -> Tuple[np.ndarray, List[Box]]:
    """
    Translate the image by (+8px, +8px) — right and down.
    Remaps all boxes by the same transform.

    Returns:
        (jittered_image, remapped_boxes)
    """
    h, w = image.shape[:2]
    jittered = _translate_image(image, _TRANSLATE_DX, _TRANSLATE_DY)
    remapped  = [remap_box_translate(b, h, w, _TRANSLATE_DX, _TRANSLATE_DY)
                 for b in boxes]
    return jittered, remapped


def apply_brightness(image: np.ndarray,
                     boxes: List[Box]) -> Tuple[np.ndarray, List[Box]]:
    """
    Increase image brightness by +10%.
    Boxes are unchanged (brightness has no geometric effect).

    Returns:
        (jittered_image, original_boxes_clipped)
    """
    h, w     = image.shape[:2]
    jittered = _brightness_image(image, _BRIGHTNESS_FACTOR)
    remapped = [remap_box_brightness(b, h, w, _BRIGHTNESS_FACTOR)
                for b in boxes]
    return jittered, remapped


def apply_jitter(image:     np.ndarray,
                 boxes:     List[Box],
                 transform: str) -> Tuple[np.ndarray, List[Box]]:
    """
    Dispatch function. Apply the named jitter transform.

    Args:
        image:     uint8 BGR image [H, W, 3]
        boxes:     list of (x1, y1, x2, y2) boxes in image pixel coords
        transform: one of 'scale', 'translate', 'brightness'

    Returns:
        (jittered_image, remapped_boxes)

    Raises:
        ValueError if transform name is not recognised
    """
    if transform == "scale":
        return apply_scale(image, boxes)
    elif transform == "translate":
        return apply_translate(image, boxes)
    elif transform == "brightness":
        return apply_brightness(image, boxes)
    else:
        raise ValueError(
            f"apply_jitter: unknown transform '{transform}'. "
            f"Must be one of: 'scale', 'translate', 'brightness'."
        )


# ── Image transform helpers ───────────────────────────────────────────────────

def _scale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale image around its center by scale_factor.
    Output has the same spatial dimensions as input — letterbox-style:
    regions that fall outside the original bounds are filled with black (0).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Affine matrix for scale around center
    M = np.array([
        [scale_factor, 0.0,          cx * (1.0 - scale_factor)],
        [0.0,          scale_factor, cy * (1.0 - scale_factor)],
    ], dtype=np.float64)

    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def _translate_image(image: np.ndarray,
                     dx: float,
                     dy: float) -> np.ndarray:
    """
    Translate image by (dx, dy) pixels.
    Regions that move out of frame are filled with black (0).
    """
    h, w = image.shape[:2]
    M = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
    ], dtype=np.float64)

    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def _brightness_image(image: np.ndarray,
                      factor: float) -> np.ndarray:
    """
    Multiply pixel values by factor, clipping to [0, 255].
    Operates in float32 to avoid uint8 overflow artefacts.
    """
    img_f = image.astype(np.float32) * factor
    return np.clip(img_f, 0, 255).astype(np.uint8)