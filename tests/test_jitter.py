"""
tests/test_jitter.py
====================
Unit tests for inference/jitter.py and the box remapping utilities.

Tests cover:
  - scale remap: box moves correctly under +3% scale around center
  - translate remap: box shifts by (+8, +8) pixels
  - brightness: boxes are unchanged (no geometric effect)
  - clipping: boxes outside image bounds are clipped to [0, img_w/h]
  - determinism: same inputs → identical outputs across repeated calls
  - dispatch: apply_jitter routes to correct transform
  - image shapes: output image has same spatial dimensions as input
  - unknown transform raises ValueError
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.inference.jitter import (
    apply_brightness,
    apply_jitter,
    apply_scale,
    apply_translate,
    _BRIGHTNESS_FACTOR,
    _SCALE_FACTOR,
    _TRANSLATE_DX,
    _TRANSLATE_DY,
)
from autolabel.utils.box_transform import SCALE_DELTA, TRANSLATE_DELTA


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_image(h=480, w=640, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ── Scale tests ───────────────────────────────────────────────────────────────

def test_scale_image_shape_preserved():
    img = make_image(h=480, w=640)
    out, _ = apply_scale(img, [])
    assert out.shape == img.shape
    print("PASS  test_scale_image_shape_preserved")


def test_scale_box_moves_away_from_center():
    """
    Scaling up (+3%) around center moves boxes away from the image center.
    A box centered at the image center should stay near center.
    A box in a corner should move further from center.
    """
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    cx, cy = w / 2.0, h / 2.0

    # Box near top-left corner
    box = (50.0, 40.0, 150.0, 120.0)
    _, remapped = apply_scale(img, [box])
    rb = remapped[0]

    # After scaling up, box should move further from center (top-left direction)
    orig_cx  = (box[0] + box[2]) / 2.0
    orig_cy  = (box[1] + box[3]) / 2.0
    new_cx   = (rb[0]  + rb[2])  / 2.0
    new_cy   = (rb[1]  + rb[3])  / 2.0

    orig_dist = ((orig_cx - cx)**2 + (orig_cy - cy)**2) ** 0.5
    new_dist  = ((new_cx  - cx)**2 + (new_cy  - cy)**2) ** 0.5

    assert new_dist > orig_dist, (
        f"Scale up should move box away from center: "
        f"orig_dist={orig_dist:.2f}, new_dist={new_dist:.2f}"
    )
    print("PASS  test_scale_box_moves_away_from_center")


def test_scale_box_center_at_image_center_stays():
    """A box centered exactly at the image center should stay at center."""
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    cx, cy = w / 2.0, h / 2.0

    # Box centered at image center
    half = 50.0
    box = (cx - half, cy - half, cx + half, cy + half)
    _, remapped = apply_scale(img, [box])
    rb = remapped[0]

    new_cx = (rb[0] + rb[2]) / 2.0
    new_cy = (rb[1] + rb[3]) / 2.0
    assert abs(new_cx - cx) < 1.0
    assert abs(new_cy - cy) < 1.0
    print("PASS  test_scale_box_center_at_image_center_stays")


def test_scale_box_size_increases():
    """Scaling up by 3% should increase box dimensions by ~3%."""
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    box  = (200.0, 150.0, 400.0, 350.0)
    _, remapped = apply_scale(img, [box])
    rb = remapped[0]

    orig_w = box[2] - box[0]
    orig_h = box[3] - box[1]
    new_w  = rb[2]  - rb[0]
    new_h  = rb[3]  - rb[1]

    expected_scale = _SCALE_FACTOR
    assert abs(new_w / orig_w - expected_scale) < 0.01
    assert abs(new_h / orig_h - expected_scale) < 0.01
    print("PASS  test_scale_box_size_increases")


# ── Translate tests ───────────────────────────────────────────────────────────

def test_translate_image_shape_preserved():
    img = make_image()
    out, _ = apply_translate(img, [])
    assert out.shape == img.shape
    print("PASS  test_translate_image_shape_preserved")


def test_translate_box_shifts_by_delta():
    """Box coordinates should shift by exactly (dx, dy)."""
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    box  = (100.0, 80.0, 300.0, 250.0)
    _, remapped = apply_translate(img, [box])
    rb = remapped[0]

    assert abs(rb[0] - (box[0] + _TRANSLATE_DX)) < 1e-6
    assert abs(rb[1] - (box[1] + _TRANSLATE_DY)) < 1e-6
    assert abs(rb[2] - (box[2] + _TRANSLATE_DX)) < 1e-6
    assert abs(rb[3] - (box[3] + _TRANSLATE_DY)) < 1e-6
    print("PASS  test_translate_box_shifts_by_delta")


def test_translate_box_clipped_at_boundary():
    """Box near right/bottom edge should be clipped after translation."""
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    # Box near bottom-right corner — will exceed bounds after +8px shift
    box  = (620.0, 465.0, 638.0, 478.0)
    _, remapped = apply_translate(img, [box])
    rb = remapped[0]

    assert rb[0] <= w, f"x1={rb[0]} exceeds width {w}"
    assert rb[1] <= h, f"y1={rb[1]} exceeds height {h}"
    assert rb[2] <= w, f"x2={rb[2]} exceeds width {w}"
    assert rb[3] <= h, f"y2={rb[3]} exceeds height {h}"
    print("PASS  test_translate_box_clipped_at_boundary")


# ── Brightness tests ──────────────────────────────────────────────────────────

def test_brightness_image_shape_preserved():
    img = make_image()
    out, _ = apply_brightness(img, [])
    assert out.shape == img.shape
    print("PASS  test_brightness_image_shape_preserved")


def test_brightness_boxes_unchanged():
    """Brightness has no geometric effect — boxes must be identical."""
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    boxes = [
        (100.0, 80.0, 300.0, 250.0),
        (400.0, 200.0, 600.0, 400.0),
    ]
    _, remapped = apply_brightness(img, boxes)

    for orig, remap in zip(boxes, remapped):
        for o, r in zip(orig, remap):
            assert abs(o - r) < 1e-6, (
                f"Brightness should not move boxes: orig={orig}, remap={remap}"
            )
    print("PASS  test_brightness_boxes_unchanged")


def test_brightness_pixels_increase():
    """Brightened image should have higher mean pixel value."""
    img = make_image(seed=7)
    out, _ = apply_brightness(img, [])
    assert out.mean() > img.mean()
    print("PASS  test_brightness_pixels_increase")


def test_brightness_clipped_to_255():
    """Bright pixels should not exceed 255 after brightening."""
    img = np.full((100, 100, 3), 250, dtype=np.uint8)
    out, _ = apply_brightness(img, [])
    assert out.max() <= 255
    assert out.dtype == np.uint8
    print("PASS  test_brightness_clipped_to_255")


# ── Clipping tests ────────────────────────────────────────────────────────────

def test_scale_clips_to_image_bounds():
    """No remapped box coordinate should be negative or exceed image dims."""
    h, w = 480, 640
    img  = make_image(h=h, w=w)
    # Box in corner — may go negative after scale (if scale < 1)
    # We use +3% scale so boxes go outward, but still test clipping contract
    boxes = [(0.0, 0.0, 50.0, 50.0), (590.0, 430.0, 639.0, 479.0)]
    _, remapped = apply_scale(img, boxes)
    for rb in remapped:
        assert rb[0] >= 0.0 and rb[0] <= w
        assert rb[1] >= 0.0 and rb[1] <= h
        assert rb[2] >= 0.0 and rb[2] <= w
        assert rb[3] >= 0.0 and rb[3] <= h
    print("PASS  test_scale_clips_to_image_bounds")


# ── Determinism tests ─────────────────────────────────────────────────────────

def test_all_transforms_deterministic():
    """Repeated calls with same inputs must give identical outputs."""
    img   = make_image(seed=42)
    boxes = [(100.0, 80.0, 300.0, 250.0), (400.0, 200.0, 550.0, 380.0)]

    for transform in ["scale", "translate", "brightness"]:
        out1, rb1 = apply_jitter(img, boxes, transform)
        out2, rb2 = apply_jitter(img, boxes, transform)

        assert np.array_equal(out1, out2), (
            f"{transform}: image output not deterministic")
        assert rb1 == rb2, (
            f"{transform}: remapped boxes not deterministic")

    print("PASS  test_all_transforms_deterministic")


# ── Dispatch tests ────────────────────────────────────────────────────────────

def test_apply_jitter_dispatch():
    """apply_jitter correctly routes to each transform."""
    img   = make_image()
    boxes = [(100.0, 80.0, 300.0, 250.0)]

    for name in ["scale", "translate", "brightness"]:
        out, rb = apply_jitter(img, boxes, name)
        assert out.shape == img.shape
        assert len(rb) == 1
    print("PASS  test_apply_jitter_dispatch")


def test_apply_jitter_unknown_raises():
    img = make_image()
    try:
        apply_jitter(img, [], "flip")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS  test_apply_jitter_unknown_raises")


def test_empty_box_list():
    """All transforms handle empty box list without error."""
    img = make_image()
    for transform in ["scale", "translate", "brightness"]:
        out, rb = apply_jitter(img, [], transform)
        assert out.shape == img.shape
        assert rb == []
    print("PASS  test_empty_box_list")


def test_multiple_boxes_all_remapped():
    """All boxes in the input list are remapped."""
    img   = make_image()
    boxes = [(i * 50.0, i * 30.0, i * 50.0 + 40.0, i * 30.0 + 30.0)
             for i in range(5)]
    for transform in ["scale", "translate", "brightness"]:
        _, rb = apply_jitter(img, boxes, transform)
        assert len(rb) == len(boxes), (
            f"{transform}: expected {len(boxes)} remapped boxes, got {len(rb)}")
    print("PASS  test_multiple_boxes_all_remapped")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_scale_image_shape_preserved,
        test_scale_box_moves_away_from_center,
        test_scale_box_center_at_image_center_stays,
        test_scale_box_size_increases,
        test_translate_image_shape_preserved,
        test_translate_box_shifts_by_delta,
        test_translate_box_clipped_at_boundary,
        test_brightness_image_shape_preserved,
        test_brightness_boxes_unchanged,
        test_brightness_pixels_increase,
        test_brightness_clipped_to_255,
        test_scale_clips_to_image_bounds,
        test_all_transforms_deterministic,
        test_apply_jitter_dispatch,
        test_apply_jitter_unknown_raises,
        test_empty_box_list,
        test_multiple_boxes_all_remapped,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    print(f"Phase 2 jitter tests: {passed} passed, {failed} failed")
    print("=" * 50)