"""
utils/iou.py
============
Vectorized IoU computation.

Rules:
  - All IoU calls in the system go through this module
  - No inline reimplementation anywhere else
  - CPU only (inputs are plain Python tuples or numpy arrays)
  - box format: (x1, y1, x2, y2) absolute pixels throughout
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np


Box  = Tuple[float, float, float, float]   # (x1, y1, x2, y2)
Boxes = np.ndarray                          # shape [N, 4]


def iou_pair(box_a: Box, box_b: Box) -> float:
    """
    IoU between two boxes given as (x1, y1, x2, y2) tuples.
    Returns 0.0 if either box has zero area or if there is no intersection.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter

    if union <= 0.0:
        return 0.0

    return float(inter / union)


def iou_matrix(boxes_a: Boxes, boxes_b: Boxes) -> np.ndarray:
    """
    Vectorized IoU between all pairs in boxes_a and boxes_b.

    Args:
        boxes_a: [N, 4] array of (x1, y1, x2, y2) boxes
        boxes_b: [M, 4] array of (x1, y1, x2, y2) boxes

    Returns:
        [N, M] float64 array of IoU values
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # Expand for broadcasting: [N,1,4] vs [1,M,4]
    a = boxes_a[:, None, :]  # [N, 1, 4]
    b = boxes_b[None, :, :]  # [1, M, 4]

    ix1 = np.maximum(a[:, :, 0], b[:, :, 0])
    iy1 = np.maximum(a[:, :, 1], b[:, :, 1])
    ix2 = np.minimum(a[:, :, 2], b[:, :, 2])
    iy2 = np.minimum(a[:, :, 3], b[:, :, 3])

    iw    = np.maximum(0.0, ix2 - ix1)
    ih    = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih                                           # [N, M]

    area_a = (np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) *
              np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1]))  # [N]
    area_b = (np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) *
              np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1]))  # [M]

    union = area_a[:, None] + area_b[None, :] - inter         # [N, M]

    # Avoid division by zero
    iou = np.where(union > 0.0, inter / union, 0.0)
    return iou.astype(np.float64)


def boxes_to_numpy(boxes: Sequence[Box]) -> np.ndarray:
    """Convert a list of (x1,y1,x2,y2) tuples to a [N,4] numpy array."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array(boxes, dtype=np.float64)