"""
utils/box_transform.py
======================
Box remapping under the three deterministic jitter transforms.

Rules:
  - All box remapping goes through this module
  - The three transforms are: scale (±3%), translation (±8px),
    brightness (±10%) — brightness does not move boxes
  - Parameters are hard-coded constants matching jitter.py
  - Input/output boxes are (x1, y1, x2, y2) absolute pixels
  - Image dimensions are required to clip boxes to valid range
  - CPU only
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Hard-coded jitter constants — must match inference/jitter.py exactly
SCALE_DELTA       = 0.03   # ±3%
TRANSLATE_DELTA   = 8      # ±8 pixels
BRIGHTNESS_DELTA  = 0.10   # ±10% — no geometric effect on boxes

Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2)


def remap_box_scale(box: Box,
                    img_h: int,
                    img_w: int,
                    scale_factor: float) -> Box:
    """
    Remap a box under a scale transform centered on the image center.

    The scale transform resizes the image by scale_factor around the image
    center. Boxes are transformed by the same scaling.

    Args:
        box:          (x1, y1, x2, y2) in original image coordinates
        img_h:        original image height in pixels
        img_w:        original image width in pixels
        scale_factor: e.g. 1.03 for +3%, 0.97 for -3%

    Returns:
        Remapped box clipped to [0, img_w] × [0, img_h].
    """
    cx = img_w / 2.0
    cy = img_h / 2.0

    x1, y1, x2, y2 = box

    # Scale around center
    nx1 = cx + (x1 - cx) * scale_factor
    ny1 = cy + (y1 - cy) * scale_factor
    nx2 = cx + (x2 - cx) * scale_factor
    ny2 = cy + (y2 - cy) * scale_factor

    return _clip_box((nx1, ny1, nx2, ny2), img_h, img_w)


def remap_box_translate(box: Box,
                        img_h: int,
                        img_w: int,
                        dx: float,
                        dy: float) -> Box:
    """
    Remap a box under a translation transform.

    Args:
        box:   (x1, y1, x2, y2) in original image coordinates
        img_h: original image height
        img_w: original image width
        dx:    horizontal shift in pixels (positive = right)
        dy:    vertical shift in pixels (positive = down)

    Returns:
        Translated box clipped to [0, img_w] × [0, img_h].
    """
    x1, y1, x2, y2 = box
    return _clip_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), img_h, img_w)


def remap_box_brightness(box: Box,
                          img_h: int,
                          img_w: int,
                          brightness_factor: float) -> Box:
    """
    Brightness transform has no geometric effect on boxes.
    Returns the original box unchanged (clipped for safety).

    Args:
        brightness_factor: ignored — included for API symmetry with other
                           remap functions so callers can use the same
                           dispatch pattern.
    """
    return _clip_box(box, img_h, img_w)


def remap_box(box: Box,
              img_h: int,
              img_w: int,
              transform: str,
              **kwargs) -> Box:
    """
    Dispatch function. Remaps box under the named transform.

    Args:
        box:       (x1, y1, x2, y2)
        img_h:     image height
        img_w:     image width
        transform: one of 'scale', 'translate', 'brightness'
        **kwargs:  transform-specific parameters:
                   scale      → scale_factor (float)
                   translate  → dx (float), dy (float)
                   brightness → brightness_factor (float)

    Returns:
        Remapped box clipped to image bounds.

    Raises:
        ValueError: if transform name is not recognised
    """
    if transform == "scale":
        return remap_box_scale(
            box, img_h, img_w,
            scale_factor=kwargs.get("scale_factor", 1.0 + SCALE_DELTA)
        )
    elif transform == "translate":
        return remap_box_translate(
            box, img_h, img_w,
            dx=kwargs.get("dx", TRANSLATE_DELTA),
            dy=kwargs.get("dy", TRANSLATE_DELTA),
        )
    elif transform == "brightness":
        return remap_box_brightness(
            box, img_h, img_w,
            brightness_factor=kwargs.get("brightness_factor",
                                         1.0 + BRIGHTNESS_DELTA)
        )
    else:
        raise ValueError(
            f"remap_box: unknown transform '{transform}'. "
            f"Must be one of: 'scale', 'translate', 'brightness'."
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clip_box(box: Box, img_h: int, img_w: int) -> Box:
    """
    Clip a box to the image boundary and ensure x1<=x2, y1<=y2.
    Returns a zero-area box at the boundary if the box is fully outside.
    """
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0.0, img_w))
    y1 = float(np.clip(y1, 0.0, img_h))
    x2 = float(np.clip(x2, 0.0, img_w))
    y2 = float(np.clip(y2, 0.0, img_h))
    # Ensure ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)