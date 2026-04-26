"""
tests/test_scorer.py
====================
Unit tests for scoring/scorer.py.

Tests cover:
  - Matched pair: all three components computed, score in [0,1]
  - Unmatched box: score=None, all components=None
  - Ambiguous box: score=None, all components=None
  - C_jitter_loc=0.0 (prefilter failed) is distinct from None (unmatched)
  - jitter_executed=False when jitter_ious absent (prefilter not passed)
  - jitter_executed=True when jitter_ious present
  - Perfect stability (identical boxes, same class, IoU jitter=1): score≈1
  - Zero stability (different class, no overlap, jitter IoU=0): score≈0
  - Score is clamped to [0,1]
  - Alpha+beta+gamma weighting is applied correctly
  - Population std (ddof=0) used for C_jitter_loc
  - ScoringResult returned (not separate scores/components dicts)
  - Every box in C_t appears in scores and components
  - Components are independent (verified individually)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import (
    CandidatePool,
    ComponentRecord,
    MatchResult,
    PseudoLabel,
    ScoringResult,
)
from autolabel.scoring.scorer import ScoringConfig, StabilityScorer


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_scores(pred_class: int, n: int = 80) -> tuple:
    s = [0.0] * n
    s[pred_class] = 1.0
    return tuple(s)


def make_pl(image_id="img", box_id=None, round_id=1,
            box=(100.0, 100.0, 200.0, 200.0),
            pred_class=0, confidence=0.8,
            class_scores=None) -> PseudoLabel:
    if box_id is None:
        box_id = f"{image_id}_r{round_id}_0000"
    if class_scores is None:
        class_scores = make_scores(pred_class)
    return PseudoLabel(
        image_id=image_id, box_id=box_id, round_id=round_id,
        box=box, pred_class=pred_class,
        class_scores=class_scores, confidence=confidence,
    )


def default_config(**kwargs) -> ScoringConfig:
    cfg = ScoringConfig()
    for k, v in kwargs.items():
        object.__setattr__(cfg, k, v)
    return cfg


def make_match_result(image_id="img", round_t=1,
                      matched_pairs=None,
                      unmatched_current=None,
                      ambiguous=None) -> MatchResult:
    return MatchResult(
        image_id=image_id, round_t=round_t,
        matched_pairs=matched_pairs or [],
        unmatched_current=unmatched_current or [],
        ambiguous=ambiguous or [],
    )


def run_scorer(current_pls, prev_pls, matched_pairs=None,
               unmatched=None, ambiguous=None,
               jitter_ious=None, config=None):
    """Helper: build pool/match_results/jitter_ious and call scorer.score()."""
    image_id = current_pls[0].image_id if current_pls else "img"
    C_t   = {image_id: current_pls}
    C_prev= {image_id: prev_pls}
    mr    = make_match_result(
        image_id=image_id,
        matched_pairs=matched_pairs or [],
        unmatched_current=unmatched or [],
        ambiguous=ambiguous or [],
    )
    match_results = {image_id: mr}
    scorer = StabilityScorer(config or ScoringConfig())
    return scorer.score(C_t, C_prev, match_results, jitter_ious or {})


# ── Return type tests ─────────────────────────────────────────────────────────

def test_returns_scoring_result():
    curr = [make_pl(box_id="t_0")]
    prev = [make_pl(box_id="p_0", round_id=0)]
    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")])
    assert isinstance(result, ScoringResult)
    assert isinstance(result.scores, dict)
    assert isinstance(result.components, dict)
    print("PASS  test_returns_scoring_result")


def test_all_ct_boxes_in_output():
    """Every box in C_t must appear in both scores and components."""
    curr = [make_pl(box_id="t_0"), make_pl(box_id="t_1"),
            make_pl(box_id="t_2")]
    prev = [make_pl(box_id="p_0", round_id=0)]
    result = run_scorer(
        curr, prev,
        matched_pairs=[("t_0", "p_0")],
        unmatched=["t_1"],
        ambiguous=["t_2"],
    )
    for pl in curr:
        assert pl.box_id in result.scores,     f"{pl.box_id} missing from scores"
        assert pl.box_id in result.components, f"{pl.box_id} missing from components"
    print("PASS  test_all_ct_boxes_in_output")


# ── Matched pair tests ────────────────────────────────────────────────────────

def test_matched_pair_score_in_range():
    curr = [make_pl(box_id="t_0", box=(100.0, 100.0, 200.0, 200.0), pred_class=0)]
    prev = [make_pl(box_id="p_0", box=(105.0, 105.0, 205.0, 205.0),
                    pred_class=0, round_id=0)]
    result = run_scorer(curr, prev, matched_pairs=[("t_0", "p_0")])
    s = result.scores["t_0"]
    assert s is not None
    assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1]"
    print("PASS  test_matched_pair_score_in_range")


def test_matched_pair_all_components_set():
    curr = [make_pl(box_id="t_0", box=(100.0, 100.0, 200.0, 200.0))]
    prev = [make_pl(box_id="p_0", box=(105.0, 105.0, 205.0, 205.0), round_id=0)]
    result = run_scorer(curr, prev, matched_pairs=[("t_0", "p_0")])
    cr = result.components["t_0"]
    assert cr.c_cls_dist  is not None
    assert cr.c_round_loc is not None
    assert cr.c_jitter_loc is not None
    print("PASS  test_matched_pair_all_components_set")


def test_perfect_stability_score_near_one():
    """
    Identical box, same class, jitter IoUs all 1.0 → all components = 1.0
    → score ≈ 1.0.
    """
    box = (100.0, 100.0, 200.0, 200.0)
    sc  = make_scores(0)
    curr = [make_pl(box_id="t_0", box=box, class_scores=sc)]
    prev = [make_pl(box_id="p_0", box=box, class_scores=sc, round_id=0)]
    result = run_scorer(
        curr, prev,
        matched_pairs=[("t_0", "p_0")],
        jitter_ious={"t_0": (1.0, 1.0, 1.0)},
    )
    s = result.scores["t_0"]
    assert abs(s - 1.0) < 1e-6, f"Expected score≈1.0, got {s}"
    cr = result.components["t_0"]
    assert abs(cr.c_cls_dist  - 1.0) < 1e-6
    assert abs(cr.c_round_loc - 1.0) < 1e-6
    assert abs(cr.c_jitter_loc- 1.0) < 1e-6
    print("PASS  test_perfect_stability_score_near_one")


def test_zero_stability_score_near_zero():
    """
    Different class (JS=1 → C_cls_dist=0), no overlap (IoU=0 → C_round_loc=0),
    jitter IoUs all 0.0 (C_jitter_loc = 1 - std([0,0,0]) = 1.0).

    Note: std([0,0,0]) = 0 → C_jitter_loc = 1-0 = 1.0 ≠ 0.
    True zero stability requires jitter absent (C_jitter_loc=0.0).
    With gamma=0.34 and C_jitter_loc=0.0: score = 0*alpha + 0*beta + 0*gamma = 0.
    """
    sc0 = make_scores(0)
    sc1 = make_scores(1)
    curr = [make_pl(box_id="t_0", box=(0.0, 0.0, 10.0, 10.0),
                    class_scores=sc0)]
    prev = [make_pl(box_id="p_0", box=(500.0, 500.0, 600.0, 600.0),
                    class_scores=sc1, round_id=0)]
    # No jitter_ious → C_jitter_loc = 0.0
    result = run_scorer(curr, prev, matched_pairs=[("t_0", "p_0")])
    s  = result.scores["t_0"]
    cr = result.components["t_0"]
    assert abs(cr.c_cls_dist  - 0.0) < 1e-6, f"c_cls_dist={cr.c_cls_dist}"
    assert abs(cr.c_round_loc - 0.0) < 1e-6, f"c_round_loc={cr.c_round_loc}"
    assert abs(cr.c_jitter_loc- 0.0) < 1e-6, f"c_jitter_loc={cr.c_jitter_loc}"
    assert abs(s - 0.0) < 1e-6, f"Expected score≈0.0, got {s}"
    print("PASS  test_zero_stability_score_near_zero")


# ── Null convention tests ─────────────────────────────────────────────────────

def test_unmatched_box_score_is_none():
    curr = [make_pl(box_id="t_0")]
    prev = []
    result = run_scorer(curr, prev, unmatched=["t_0"])
    assert result.scores["t_0"] is None
    print("PASS  test_unmatched_box_score_is_none")


def test_unmatched_box_components_all_none():
    curr = [make_pl(box_id="t_0")]
    prev = []
    result = run_scorer(curr, prev, unmatched=["t_0"])
    cr = result.components["t_0"]
    assert cr.c_cls_dist   is None
    assert cr.c_round_loc  is None
    assert cr.c_jitter_loc is None
    assert cr.jitter_executed == False
    print("PASS  test_unmatched_box_components_all_none")


def test_ambiguous_box_score_is_none():
    curr = [make_pl(box_id="t_0")]
    prev = [make_pl(box_id="p_0", round_id=0),
            make_pl(box_id="p_1", round_id=0)]
    result = run_scorer(curr, prev, ambiguous=["t_0"])
    assert result.scores["t_0"] is None
    print("PASS  test_ambiguous_box_score_is_none")


def test_ambiguous_box_components_all_none():
    curr = [make_pl(box_id="t_0")]
    prev = [make_pl(box_id="p_0", round_id=0)]
    result = run_scorer(curr, prev, ambiguous=["t_0"])
    cr = result.components["t_0"]
    assert cr.c_cls_dist   is None
    assert cr.c_round_loc  is None
    assert cr.c_jitter_loc is None
    assert cr.jitter_executed == False
    print("PASS  test_ambiguous_box_components_all_none")


# ── Jitter distinction tests ──────────────────────────────────────────────────

def test_jitter_executed_false_when_absent():
    """C_jitter_loc=0.0 and jitter_executed=False when box absent from jitter_ious."""
    curr = [make_pl(box_id="t_0", box=(100.0, 100.0, 200.0, 200.0))]
    prev = [make_pl(box_id="p_0", box=(100.0, 100.0, 200.0, 200.0), round_id=0)]
    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")],
                        jitter_ious={})  # explicitly empty
    cr = result.components["t_0"]
    assert cr.c_jitter_loc   == 0.0,  f"Expected 0.0, got {cr.c_jitter_loc}"
    assert cr.jitter_executed == False
    print("PASS  test_jitter_executed_false_when_absent")


def test_jitter_executed_true_when_present():
    """C_jitter_loc computed and jitter_executed=True when triple present."""
    box = (100.0, 100.0, 200.0, 200.0)
    curr = [make_pl(box_id="t_0", box=box)]
    prev = [make_pl(box_id="p_0", box=box, round_id=0)]
    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")],
                        jitter_ious={"t_0": (0.9, 0.85, 0.88)})
    cr = result.components["t_0"]
    assert cr.jitter_executed == True
    assert cr.c_jitter_loc    is not None
    assert cr.c_jitter_loc    != 0.0
    print("PASS  test_jitter_executed_true_when_present")


def test_jitter_zero_vs_none_distinct():
    """
    C_jitter_loc=0.0 (prefilter failed) is DISTINCT from None (unmatched).
    This is the core null convention from the spec.
    """
    box  = (100.0, 100.0, 200.0, 200.0)
    curr = [make_pl(box_id="t_matched",   box=box),
            make_pl(box_id="t_unmatched", box=box)]
    prev = [make_pl(box_id="p_0", box=box, round_id=0)]

    result = run_scorer(
        curr, prev,
        matched_pairs=[("t_matched", "p_0")],
        unmatched=["t_unmatched"],
        jitter_ious={},  # t_matched prefilter not passed → 0.0
    )
    cr_matched   = result.components["t_matched"]
    cr_unmatched = result.components["t_unmatched"]

    assert cr_matched.c_jitter_loc   == 0.0,  "Matched+no-jitter should be 0.0"
    assert cr_unmatched.c_jitter_loc is None, "Unmatched should be None"
    assert cr_matched.c_jitter_loc != cr_unmatched.c_jitter_loc
    print("PASS  test_jitter_zero_vs_none_distinct")


# ── C_jitter_loc formula tests ────────────────────────────────────────────────

def test_c_jitter_loc_population_std():
    """
    C_jitter_loc = 1 - population_std([iou1, iou2, iou3]) with ddof=0.
    Verify formula against known values.
    """
    box = (100.0, 100.0, 200.0, 200.0)
    curr = [make_pl(box_id="t_0", box=box)]
    prev = [make_pl(box_id="p_0", box=box, round_id=0)]
    iou_triple = (0.9, 0.8, 0.7)
    expected_std = float(np.std(list(iou_triple), ddof=0))
    expected_jitter = 1.0 - expected_std

    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")],
                        jitter_ious={"t_0": iou_triple})
    cr = result.components["t_0"]
    assert abs(cr.c_jitter_loc - expected_jitter) < 1e-9, (
        f"Expected c_jitter_loc={expected_jitter:.6f}, got {cr.c_jitter_loc:.6f}")
    print("PASS  test_c_jitter_loc_population_std")


def test_c_jitter_loc_identical_ious():
    """Identical jitter IoUs → std=0 → C_jitter_loc=1.0."""
    box = (100.0, 100.0, 200.0, 200.0)
    curr = [make_pl(box_id="t_0", box=box)]
    prev = [make_pl(box_id="p_0", box=box, round_id=0)]
    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")],
                        jitter_ious={"t_0": (0.8, 0.8, 0.8)})
    cr = result.components["t_0"]
    assert abs(cr.c_jitter_loc - 1.0) < 1e-9, (
        f"Identical IoUs → C_jitter_loc should be 1.0, got {cr.c_jitter_loc}")
    print("PASS  test_c_jitter_loc_identical_ious")


# ── Weighting tests ───────────────────────────────────────────────────────────

def test_score_weighting():
    """
    S = alpha*C_cls_dist + beta*C_round_loc + gamma*C_jitter_loc.
    Use known component values to verify weighting.
    """
    # Identical boxes and class scores → C_cls_dist=1, C_round_loc=1
    # Jitter IoUs all 0.5 → std=0 → C_jitter_loc=1
    box = (100.0, 100.0, 200.0, 200.0)
    sc  = make_scores(0)
    cfg = ScoringConfig(alpha=0.5, beta=0.3, gamma=0.2)
    curr = [make_pl(box_id="t_0", box=box, class_scores=sc)]
    prev = [make_pl(box_id="p_0", box=box, class_scores=sc, round_id=0)]
    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")],
                        jitter_ious={"t_0": (0.5, 0.5, 0.5)},
                        config=cfg)
    # All components = 1.0 → score = 0.5*1 + 0.3*1 + 0.2*1 = 1.0
    s = result.scores["t_0"]
    assert abs(s - 1.0) < 1e-6, f"Expected 1.0, got {s}"

    # Now: C_cls_dist=1, C_round_loc=1, C_jitter_loc=0.0 (no jitter)
    # score = 0.5*1 + 0.3*1 + 0.2*0 = 0.8
    result2 = run_scorer(curr, prev,
                         matched_pairs=[("t_0", "p_0")],
                         jitter_ious={},
                         config=cfg)
    s2 = result2.scores["t_0"]
    assert abs(s2 - 0.8) < 1e-6, f"Expected 0.8, got {s2}"
    print("PASS  test_score_weighting")


def test_score_clamped_to_zero_one():
    """Score must be clamped to [0,1] even if weighted sum exceeds bounds."""
    # All components = 1.0, weights sum > 1 → raw sum > 1 → clamp to 1
    box = (100.0, 100.0, 200.0, 200.0)
    sc  = make_scores(0)
    cfg = ScoringConfig(alpha=0.5, beta=0.5, gamma=0.5)  # sums to 1.5
    curr = [make_pl(box_id="t_0", box=box, class_scores=sc)]
    prev = [make_pl(box_id="p_0", box=box, class_scores=sc, round_id=0)]
    result = run_scorer(curr, prev,
                        matched_pairs=[("t_0", "p_0")],
                        jitter_ious={"t_0": (1.0, 1.0, 1.0)},
                        config=cfg)
    s = result.scores["t_0"]
    assert s <= 1.0, f"Score {s} exceeds 1.0"
    assert s >= 0.0, f"Score {s} below 0.0"
    print("PASS  test_score_clamped_to_zero_one")


# ── Multi-image tests ─────────────────────────────────────────────────────────

def test_multiple_images():
    """Scorer handles multiple images in pool correctly."""
    box = (100.0, 100.0, 200.0, 200.0)
    sc  = make_scores(0)

    C_t = {
        "img_a": [make_pl(image_id="img_a", box_id="a_t_0", box=box, class_scores=sc)],
        "img_b": [make_pl(image_id="img_b", box_id="b_t_0", box=box, class_scores=sc)],
    }
    C_prev = {
        "img_a": [make_pl(image_id="img_a", box_id="a_p_0", box=box,
                          class_scores=sc, round_id=0)],
        "img_b": [make_pl(image_id="img_b", box_id="b_p_0", box=box,
                          class_scores=sc, round_id=0)],
    }
    match_results = {
        "img_a": make_match_result("img_a", matched_pairs=[("a_t_0", "a_p_0")]),
        "img_b": make_match_result("img_b", matched_pairs=[("b_t_0", "b_p_0")]),
    }
    scorer = StabilityScorer(ScoringConfig())
    result = scorer.score(C_t, C_prev, match_results,
                          jitter_ious={"a_t_0": (1.0, 1.0, 1.0),
                                       "b_t_0": (1.0, 1.0, 1.0)})
    assert "a_t_0" in result.scores and "b_t_0" in result.scores
    assert result.scores["a_t_0"] is not None
    assert result.scores["b_t_0"] is not None
    print("PASS  test_multiple_images")



# ── Error handling tests ───────────────────────────────────────────────────────────────────────────

def test_inconsistent_match_results_raises():
    """Fix 2: box_id in matched_pair absent from C_t must raise ValueError."""
    curr = [make_pl(box_id="t_0")]
    prev = [make_pl(box_id="p_0", round_id=0)]
    C_t    = {"img": curr}
    C_prev = {"img": prev}
    mr     = make_match_result("img", matched_pairs=[("t_GHOST", "p_0")])
    scorer = StabilityScorer(ScoringConfig())
    try:
        scorer.score(C_t, C_prev, {"img": mr}, {})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "t_GHOST" in str(e)
    print("PASS  test_inconsistent_match_results_raises")


def test_missing_image_in_match_results_raises():
    """Fix 3: C_t image absent from match_results must raise ValueError."""
    box = (100.0, 100.0, 200.0, 200.0)
    C_t = {
        "img_a": [make_pl(image_id="img_a", box_id="a_t_0", box=box)],
        "img_b": [make_pl(image_id="img_b", box_id="b_t_0", box=box)],
    }
    C_prev = {"img_a": [], "img_b": []}
    match_results = {
        "img_a": make_match_result("img_a", unmatched_current=["a_t_0"]),
    }
    scorer = StabilityScorer(ScoringConfig())
    try:
        scorer.score(C_t, C_prev, match_results, {})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "b_t_0" in str(e) or "img_b" in str(e)
    print("PASS  test_missing_image_in_match_results_raises")


# ── Runner ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_returns_scoring_result,
        test_all_ct_boxes_in_output,
        test_matched_pair_score_in_range,
        test_matched_pair_all_components_set,
        test_perfect_stability_score_near_one,
        test_zero_stability_score_near_zero,
        test_unmatched_box_score_is_none,
        test_unmatched_box_components_all_none,
        test_ambiguous_box_score_is_none,
        test_ambiguous_box_components_all_none,
        test_jitter_executed_false_when_absent,
        test_jitter_executed_true_when_present,
        test_jitter_zero_vs_none_distinct,
        test_c_jitter_loc_population_std,
        test_c_jitter_loc_identical_ious,
        test_score_weighting,
        test_score_clamped_to_zero_one,
        test_multiple_images,
        test_inconsistent_match_results_raises,
        test_missing_image_in_match_results_raises,
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
    print(f"Phase 4 scorer tests: {passed} passed, {failed} failed")
    print("=" * 50)