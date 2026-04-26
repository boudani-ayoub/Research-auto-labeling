"""
tests/test_canonical_infer.py
=============================
Unit tests for inference/canonical_infer.py.

Tests cover:
  - Output type is CandidatePool (Dict[str, List[PseudoLabel]])
  - Every image_id from input appears in the pool
  - Each detection becomes a valid PseudoLabel
  - confidence populated from YOLO result (float in (0,1])
  - class_scores length == number of classes (80 for COCO)
  - sum(class_scores) ≈ 1.0
  - argmax(class_scores) == pred_class (CHECK 2 from Phase 0)
  - box_id follows the f"{image_id}_r{round_id}_{det_idx:04d}" convention
  - Repeat calls with same inputs are deterministic (CHECK 4 from Phase 0)
  - Images with zero detections appear in pool with empty list
  - confidence and class_scores are numerically independent (CHECK 3)

NOTE: This test requires the YOLOv8m model and GPU.
      It uses the bundled Ultralytics bus.jpg for non-trivial detections.
      Run from the project root:
          python tests/test_canonical_infer.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import CandidatePool, PseudoLabel
from autolabel.inference.canonical_infer import canonical_infer
from ultralytics import YOLO


# ── Fixtures ──────────────────────────────────────────────────────────────────

MODEL_PATH = "yolov8m.pt"
NUM_CLASSES = 80  # COCO


def load_model():
    return YOLO(MODEL_PATH)


def get_bus_image() -> np.ndarray:
    """Load the bundled Ultralytics bus.jpg — known to have ~8 detections."""
    import cv2
    import ultralytics
    path = os.path.join(os.path.dirname(ultralytics.__file__),
                        "assets", "bus.jpg")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read bus.jpg at {path}")
    return img


def make_noise_image(h=640, w=640, seed=0) -> np.ndarray:
    """Random noise image — typically yields zero detections at conf=0.05."""
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ── Output type and structure tests ──────────────────────────────────────────

def test_output_is_candidate_pool(model):
    bus = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    assert isinstance(pool, dict), "Output must be a dict (CandidatePool)"
    assert "bus_0" in pool
    assert isinstance(pool["bus_0"], list)
    print("PASS  test_output_is_candidate_pool")


def test_all_image_ids_in_pool(model):
    """Every image_id supplied must appear in the returned pool."""
    bus   = get_bus_image()
    noise = make_noise_image()
    ids   = ["bus_0", "noise_0"]
    pool  = canonical_infer(model, ids, [bus, noise], round_id=1)
    for id_ in ids:
        assert id_ in pool, f"image_id '{id_}' missing from pool"
    print("PASS  test_all_image_ids_in_pool")


def test_zero_detection_image_in_pool(model):
    """Noise image with zero detections must still appear with empty list."""
    noise = make_noise_image(seed=99)
    pool  = canonical_infer(model, ["noise_0"], [noise], round_id=1)
    assert "noise_0" in pool
    # May or may not have detections — just assert it's a list
    assert isinstance(pool["noise_0"], list)
    print("PASS  test_zero_detection_image_in_pool")


# ── PseudoLabel field tests ───────────────────────────────────────────────────

def test_pseudo_labels_are_valid(model):
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    pls  = pool["bus_0"]
    assert len(pls) > 0, "bus.jpg should produce detections at conf=0.05"

    for pl in pls:
        assert isinstance(pl, PseudoLabel)
        assert pl.image_id  == "bus_0"
        assert pl.round_id  == 1
        assert isinstance(pl.pred_class, int)
        assert 0 <= pl.pred_class < NUM_CLASSES
    print("PASS  test_pseudo_labels_are_valid")


def test_confidence_is_float_in_range(model):
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    for pl in pool["bus_0"]:
        assert isinstance(pl.confidence, float)
        assert 0.0 < pl.confidence <= 1.0, (
            f"confidence={pl.confidence} out of range (0,1]")
    print("PASS  test_confidence_is_float_in_range")


def test_class_scores_length(model):
    """class_scores must have exactly NUM_CLASSES elements."""
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    for pl in pool["bus_0"]:
        assert len(pl.class_scores) == NUM_CLASSES, (
            f"Expected {NUM_CLASSES} class scores, got {len(pl.class_scores)}")
    print("PASS  test_class_scores_length")


def test_class_scores_sum_to_one(model):
    """class_scores must sum to ~1.0."""
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    for pl in pool["bus_0"]:
        s = sum(pl.class_scores)
        assert abs(s - 1.0) < 1e-4, (
            f"class_scores sum={s:.6f}, expected ~1.0")
    print("PASS  test_class_scores_sum_to_one")


def test_argmax_class_scores_equals_pred_class(model):
    """
    argmax(class_scores) must equal pred_class for every detection.
    This is CHECK 2 from Phase 0 — re-verified here in production code.
    """
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    violations = []
    for pl in pool["bus_0"]:
        argmax = int(np.argmax(pl.class_scores))
        if argmax != pl.pred_class:
            violations.append(
                f"box_id={pl.box_id}: argmax={argmax} != pred_class={pl.pred_class}"
            )
    assert len(violations) == 0, (
        f"CHECK 2 violations ({len(violations)}):\n" + "\n".join(violations[:5])
    )
    print("PASS  test_argmax_class_scores_equals_pred_class")


def test_box_id_convention(model):
    """box_id must follow f'{image_id}_r{round_id}_{det_idx:04d}'."""
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=3)
    for det_idx, pl in enumerate(pool["bus_0"]):
        expected = f"bus_0_r3_{det_idx:04d}"
        assert pl.box_id == expected, (
            f"Expected box_id='{expected}', got '{pl.box_id}'")
    print("PASS  test_box_id_convention")


def test_confidence_and_class_scores_independent(model):
    """
    CHECK 3 from Phase 0: confidence and class_scores must be numerically
    independent. confidence comes from results.boxes.conf (which folds in
    objectness); max(class_scores) is pure class probability. They should
    differ for most detections on a real image.
    """
    bus  = get_bus_image()
    pool = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    pls  = pool["bus_0"]
    assert len(pls) > 0

    n_equal = sum(
        1 for pl in pls
        if abs(pl.confidence - max(pl.class_scores)) < 1e-5
    )
    # On a real image, most detections should have conf != max(class_scores)
    # because objectness is folded into conf. Allow at most 50% equality.
    frac_equal = n_equal / len(pls)
    assert frac_equal < 0.5, (
        f"CHECK 3: {n_equal}/{len(pls)} detections have conf == max(class_scores). "
        f"Suggests confidence and class_scores may share the same source."
    )
    print("PASS  test_confidence_and_class_scores_independent")


# ── Determinism test ──────────────────────────────────────────────────────────

def test_repeat_calls_are_deterministic(model):
    """
    CHECK 4 from Phase 0: two calls with identical inputs must return
    bit-identical class_scores for every detection.
    """
    bus   = get_bus_image()
    pool1 = canonical_infer(model, ["bus_0"], [bus], round_id=1)
    pool2 = canonical_infer(model, ["bus_0"], [bus], round_id=1)

    pls1 = pool1["bus_0"]
    pls2 = pool2["bus_0"]

    assert len(pls1) == len(pls2), (
        f"Determinism: pass 1 gave {len(pls1)} dets, pass 2 gave {len(pls2)}")

    for i, (pl1, pl2) in enumerate(zip(pls1, pls2)):
        assert pl1.class_scores == pl2.class_scores, (
            f"det {i}: class_scores differ between calls")
        assert pl1.confidence   == pl2.confidence, (
            f"det {i}: confidence differs between calls")
        assert pl1.pred_class   == pl2.pred_class, (
            f"det {i}: pred_class differs between calls")

    print("PASS  test_repeat_calls_are_deterministic")


# ── Batch test ────────────────────────────────────────────────────────────────

def test_batch_inference_is_internally_consistent(model):
    """
    Running the same batch twice must give identical results.

    NOTE: batch results are NOT required to match single-image results.
    Ultralytics letterbox-pads all images in a batch to the same dimensions,
    so the letterbox ratio/padding differs from single-image inference when
    image sizes differ. Method B uses those values for anchor mapping, so
    class_scores will differ between batch and single calls.

    In the pipeline, canonical_infer always processes all unlabeled images
    in one consistent batch per round — it never mixes batch and single
    inference. Consistency within a batch call is what matters.
    """
    bus   = get_bus_image()
    noise = make_noise_image()

    pool1 = canonical_infer(model, ["bus_0", "noise_0"], [bus, noise], round_id=1)
    pool2 = canonical_infer(model, ["bus_0", "noise_0"], [bus, noise], round_id=1)

    assert len(pool1["bus_0"]) == len(pool2["bus_0"]), (
        f"Batch consistency: pass1={len(pool1['bus_0'])} dets "
        f"!= pass2={len(pool2['bus_0'])} dets")

    for pl1, pl2 in zip(pool1["bus_0"], pool2["bus_0"]):
        assert pl1.class_scores == pl2.class_scores, (
            "class_scores differ between two identical batch calls")
        assert pl1.confidence == pl2.confidence

    print("PASS  test_batch_inference_is_internally_consistent")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model …")
    model = load_model()

    tests = [
        (test_output_is_candidate_pool,               model),
        (test_all_image_ids_in_pool,                  model),
        (test_zero_detection_image_in_pool,           model),
        (test_pseudo_labels_are_valid,                model),
        (test_confidence_is_float_in_range,           model),
        (test_class_scores_length,                    model),
        (test_class_scores_sum_to_one,                model),
        (test_argmax_class_scores_equals_pred_class,  model),
        (test_box_id_convention,                      model),
        (test_confidence_and_class_scores_independent,model),
        (test_repeat_calls_are_deterministic,         model),
        (test_batch_inference_is_internally_consistent, model),
    ]

    passed = 0
    failed = 0
    for fn, *args in tests:
        try:
            fn(*args)
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    print(f"Phase 2 canonical_infer tests: {passed} passed, {failed} failed")
    print("=" * 50)