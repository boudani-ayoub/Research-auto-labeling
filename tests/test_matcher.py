"""
tests/test_matcher.py
=====================
Unit tests for matching/matcher.py.

Tests cover:
  - Empty current → all previous unmatched, empty result
  - Empty previous → all current unmatched
  - Perfect match (identical boxes and class scores)
  - Hard-gate rejection (IoU below min_iou_gate)
  - Ambiguity rejection (cost gap < epsilon_match)
  - Ambiguous boxes go to .ambiguous, not .unmatched_current
  - Ambiguous current boxes leave nearest prev boxes as removed
    (conservative accounting — Patch 4)
  - Matched pairs are (box_id_t, box_id_prev) tuples
  - lambda_cls=0 gives IoU-only matching
  - Cost structure: high-IoU same-class pair wins over low-IoU different-class
  - Every current box appears in exactly one of:
    matched_pairs, unmatched_current, or ambiguous
  - MatchResult is frozen (immutable)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import MatchResult, PseudoLabel
from autolabel.matching.matcher import HungarianMatcher, MatchingConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_scores(pred_class: int, n_classes: int = 80) -> tuple:
    """One-hot class score vector."""
    scores = [0.0] * n_classes
    scores[pred_class] = 1.0
    return tuple(scores)


def make_pl(image_id="img_0",
            box_id=None,
            round_id=1,
            box=(100.0, 100.0, 200.0, 200.0),
            pred_class=0,
            class_scores=None,
            confidence=0.8) -> PseudoLabel:
    if box_id is None:
        box_id = f"{image_id}_r{round_id}_auto"
    if class_scores is None:
        class_scores = make_scores(pred_class)
    return PseudoLabel(
        image_id    = image_id,
        box_id      = box_id,
        round_id    = round_id,
        box         = box,
        pred_class  = pred_class,
        class_scores= class_scores,
        confidence  = confidence,
    )


def default_config(**kwargs) -> MatchingConfig:
    cfg = MatchingConfig()
    for k, v in kwargs.items():
        object.__setattr__(cfg, k, v)
    return cfg


def all_current_accounted(result: MatchResult,
                           current: list) -> bool:
    """Every current box_id appears in exactly one of the three output lists."""
    matched  = {bt for bt, _ in result.matched_pairs}
    unmatched = set(result.unmatched_current)
    ambiguous = set(result.ambiguous)

    # Disjoint
    assert matched & unmatched == set(), "matched ∩ unmatched not empty"
    assert matched & ambiguous == set(), "matched ∩ ambiguous not empty"
    assert unmatched & ambiguous == set(), "unmatched ∩ ambiguous not empty"

    # Complete
    all_ids = {pl.box_id for pl in current}
    covered = matched | unmatched | ambiguous
    assert covered == all_ids, (
        f"Not all current boxes accounted for. "
        f"Missing: {all_ids - covered}, Extra: {covered - all_ids}"
    )
    return True


# ── Edge case tests ───────────────────────────────────────────────────────────

def test_empty_current():
    matcher = HungarianMatcher(MatchingConfig())
    prev = [make_pl(box_id="prev_0", round_id=0)]
    result = matcher.match(current=[], previous=prev,
                           image_id="img_0", round_t=1)
    assert result.matched_pairs     == []
    assert result.unmatched_current == []
    assert result.ambiguous         == []
    print("PASS  test_empty_current")


def test_empty_previous():
    matcher = HungarianMatcher(MatchingConfig())
    cur = [make_pl(box_id="cur_0", round_id=1),
           make_pl(box_id="cur_1", round_id=1)]
    result = matcher.match(current=cur, previous=[],
                           image_id="img_0", round_t=1)
    assert result.matched_pairs     == []
    assert set(result.unmatched_current) == {"cur_0", "cur_1"}
    assert result.ambiguous         == []
    assert all_current_accounted(result, cur)
    print("PASS  test_empty_previous")


def test_both_empty():
    matcher = HungarianMatcher(MatchingConfig())
    result = matcher.match(current=[], previous=[],
                           image_id="img_0", round_t=1)
    assert result.matched_pairs     == []
    assert result.unmatched_current == []
    assert result.ambiguous         == []
    print("PASS  test_both_empty")


# ── Perfect match tests ───────────────────────────────────────────────────────

def test_perfect_match_single_pair():
    """Identical box and class scores → single matched pair, IoU=1."""
    matcher = HungarianMatcher(MatchingConfig())
    box = (100.0, 100.0, 200.0, 200.0)
    cur  = [make_pl(box_id="cur_0",  round_id=1, box=box, pred_class=3)]
    prev = [make_pl(box_id="prev_0", round_id=0, box=box, pred_class=3)]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0]   == ("cur_0", "prev_0")
    assert result.unmatched_current  == []
    assert result.ambiguous          == []
    assert all_current_accounted(result, cur)
    print("PASS  test_perfect_match_single_pair")


def test_perfect_match_multiple_pairs():
    """N identical pairs — all matched, none ambiguous."""
    matcher = HungarianMatcher(MatchingConfig())
    boxes = [
        (100.0, 100.0, 200.0, 200.0),
        (300.0, 300.0, 400.0, 400.0),
        (500.0, 100.0, 600.0, 200.0),
    ]
    cur  = [make_pl(box_id=f"cur_{i}",  round_id=1, box=b, pred_class=i)
            for i, b in enumerate(boxes)]
    prev = [make_pl(box_id=f"prev_{i}", round_id=0, box=b, pred_class=i)
            for i, b in enumerate(boxes)]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert len(result.matched_pairs) == 3
    assert result.unmatched_current  == []
    assert result.ambiguous          == []
    # Each cur box matched to its corresponding prev box
    matched_dict = dict(result.matched_pairs)
    for i in range(3):
        assert matched_dict[f"cur_{i}"] == f"prev_{i}"
    assert all_current_accounted(result, cur)
    print("PASS  test_perfect_match_multiple_pairs")


# ── Hard-gate tests ───────────────────────────────────────────────────────────

def test_hard_gate_rejects_low_iou():
    """Pair with IoU < min_iou_gate must not appear in matched_pairs."""
    cfg     = MatchingConfig(min_iou_gate=0.3)
    matcher = HungarianMatcher(cfg)

    # Non-overlapping boxes → IoU = 0 → below gate
    cur  = [make_pl(box_id="cur_0",  round_id=1,
                    box=(0.0, 0.0, 50.0, 50.0),    pred_class=0)]
    prev = [make_pl(box_id="prev_0", round_id=0,
                    box=(200.0, 200.0, 300.0, 300.0), pred_class=0)]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert result.matched_pairs     == []
    assert result.unmatched_current == ["cur_0"]
    assert result.ambiguous         == []
    assert all_current_accounted(result, cur)
    print("PASS  test_hard_gate_rejects_low_iou")


def test_hard_gate_passes_sufficient_iou():
    """Pair with IoU >= min_iou_gate must be in matched_pairs."""
    cfg     = MatchingConfig(min_iou_gate=0.3)
    matcher = HungarianMatcher(cfg)

    # Overlapping boxes with IoU ≈ 0.64 (> 0.3)
    cur  = [make_pl(box_id="cur_0",  round_id=1,
                    box=(100.0, 100.0, 200.0, 200.0), pred_class=0)]
    prev = [make_pl(box_id="prev_0", round_id=0,
                    box=(120.0, 120.0, 220.0, 220.0), pred_class=0)]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0][0] == "cur_0"
    assert all_current_accounted(result, cur)
    print("PASS  test_hard_gate_passes_sufficient_iou")


# ── Ambiguity tests ───────────────────────────────────────────────────────────

def test_ambiguity_rejection_identical_prev_boxes():
    """
    Two identical previous boxes create near-zero cost gap for the current
    box → current box must be marked ambiguous, not matched or unmatched.
    """
    cfg     = MatchingConfig(min_iou_gate=0.1, epsilon_match=0.5)
    matcher = HungarianMatcher(cfg)

    box = (100.0, 100.0, 200.0, 200.0)
    cur  = [make_pl(box_id="cur_0", round_id=1,
                    box=box, pred_class=0)]
    # Two identical previous boxes — costs will be identical, gap = 0 < 0.5
    prev = [
        make_pl(box_id="prev_0", round_id=0, box=box, pred_class=0),
        make_pl(box_id="prev_1", round_id=0, box=box, pred_class=0),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert result.matched_pairs     == [], f"Expected no match, got {result.matched_pairs}"
    assert result.unmatched_current == [], f"Expected not unmatched, got {result.unmatched_current}"
    assert result.ambiguous         == ["cur_0"]
    assert all_current_accounted(result, cur)
    print("PASS  test_ambiguity_rejection_identical_prev_boxes")


def test_ambiguous_box_not_in_unmatched():
    """Ambiguous box must be in .ambiguous, not in .unmatched_current."""
    cfg     = MatchingConfig(min_iou_gate=0.1, epsilon_match=1.0)
    matcher = HungarianMatcher(cfg)

    box = (100.0, 100.0, 200.0, 200.0)
    cur  = [make_pl(box_id="cur_0", round_id=1, box=box, pred_class=0)]
    prev = [
        make_pl(box_id="prev_0", round_id=0, box=box, pred_class=0),
        make_pl(box_id="prev_1", round_id=0, box=box, pred_class=0),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert "cur_0" not in result.unmatched_current
    assert "cur_0" in result.ambiguous
    print("PASS  test_ambiguous_box_not_in_unmatched")


def test_ambiguous_prev_boxes_counted_as_removed():
    """
    Patch 4: ambiguous current boxes are NOT in matched_pairs.
    Their nearest prev boxes are therefore absent from matched_pairs
    and must be counted as removed in RawChurn.
    This test verifies the MatchResult structure supports that accounting.
    """
    cfg     = MatchingConfig(min_iou_gate=0.1, epsilon_match=1.0)
    matcher = HungarianMatcher(cfg)

    box = (100.0, 100.0, 200.0, 200.0)
    cur  = [make_pl(box_id="cur_0", round_id=1, box=box, pred_class=0)]
    prev = [
        make_pl(box_id="prev_0", round_id=0, box=box, pred_class=0),
        make_pl(box_id="prev_1", round_id=0, box=box, pred_class=0),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)

    # Neither prev box appears in matched_pairs
    matched_prev = {bp for _, bp in result.matched_pairs}
    assert "prev_0" not in matched_prev
    assert "prev_1" not in matched_prev

    # Stopping evaluator will count both prev boxes as removed
    # (they are not in matched_pairs → conservative overestimate of churn)
    print("PASS  test_ambiguous_prev_boxes_counted_as_removed")


def test_clear_match_not_ambiguous():
    """
    When one previous box is a much better match, gap is large → not ambiguous.
    """
    cfg     = MatchingConfig(min_iou_gate=0.1, epsilon_match=0.05)
    matcher = HungarianMatcher(cfg)

    box_cur  = (100.0, 100.0, 200.0, 200.0)
    # prev_0 overlaps well, prev_1 is far away
    prev_close = (105.0, 105.0, 205.0, 205.0)
    prev_far   = (500.0, 500.0, 600.0, 600.0)

    cur  = [make_pl(box_id="cur_0",   round_id=1, box=box_cur,   pred_class=0)]
    prev = [
        make_pl(box_id="prev_0", round_id=0, box=prev_close, pred_class=0),
        make_pl(box_id="prev_1", round_id=0, box=prev_far,   pred_class=0),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0][0] == "cur_0"
    assert result.ambiguous           == []
    assert all_current_accounted(result, cur)
    print("PASS  test_clear_match_not_ambiguous")


# ── Cost structure tests ──────────────────────────────────────────────────────

def test_lambda_cls_zero_gives_iou_only_matching():
    """With lambda_cls=0, cost = 1 - IoU, so class scores have no effect."""
    cfg     = MatchingConfig(min_iou_gate=0.1, lambda_cls=0.0)
    matcher = HungarianMatcher(cfg)

    box_a = (100.0, 100.0, 200.0, 200.0)
    box_b = (105.0, 105.0, 205.0, 205.0)  # high IoU with box_a

    # cur_0 and prev_0 have identical boxes but different classes
    # cur_0 and prev_1 have different boxes
    cur  = [make_pl(box_id="cur_0", round_id=1, box=box_a, pred_class=0)]
    prev = [
        make_pl(box_id="prev_0", round_id=0, box=box_b, pred_class=5),  # diff class
        make_pl(box_id="prev_1", round_id=0,
                box=(500.0, 500.0, 600.0, 600.0), pred_class=0),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    # With lambda_cls=0, should match cur_0 to prev_0 (higher IoU)
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0] == ("cur_0", "prev_0")
    print("PASS  test_lambda_cls_zero_gives_iou_only_matching")


def test_class_score_affects_matching():
    """
    With lambda_cls > 0, same-class pairs should be preferred over
    different-class pairs when IoU is similar.
    """
    cfg     = MatchingConfig(min_iou_gate=0.1, lambda_cls=1.0)
    matcher = HungarianMatcher(cfg)

    box_a = (100.0, 100.0, 200.0, 200.0)
    box_b = (102.0, 102.0, 202.0, 202.0)

    cur  = [make_pl(box_id="cur_0", round_id=1, box=box_a, pred_class=3)]
    prev = [
        make_pl(box_id="prev_same", round_id=0, box=box_b, pred_class=3),
        make_pl(box_id="prev_diff", round_id=0, box=box_b, pred_class=7),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    # Must have exactly one match — if the matcher returns zero that is a bug
    assert len(result.matched_pairs) == 1, (
        f"Expected exactly 1 match, got {len(result.matched_pairs)}: "
        f"{result.matched_pairs}")
    assert result.matched_pairs[0] == ("cur_0", "prev_same"), (
        f"Expected match with same-class prev, got {result.matched_pairs[0]}")
    print("PASS  test_class_score_affects_matching")


def test_gate_before_hungarian_saves_valid_match():
    """
    Regression test for Fix 1: gate applied BEFORE Hungarian.

    Setup:
      curr_0 has two candidate previous boxes:
        prev_bad:   low IoU (below gate) but zero JS cost (same class, same box center)
        prev_good:  high IoU (above gate) but nonzero JS cost (different class)

    Without pre-gating, Hungarian may pick prev_bad (lower cost), then
    hard-gate filtering removes it, leaving curr_0 spuriously unmatched
    even though prev_good was a valid match.

    With pre-gating, prev_bad is blocked before Hungarian runs, so
    Hungarian assigns curr_0 → prev_good correctly.
    """
    cfg     = MatchingConfig(min_iou_gate=0.3, lambda_cls=1.0, epsilon_match=0.01)
    matcher = HungarianMatcher(cfg)

    sc_cls0 = tuple([1.0] + [0.0] * 79)
    sc_cls1 = tuple([0.0, 1.0] + [0.0] * 78)

    curr = [make_pl(box_id="curr_0", box=(100.0, 100.0, 200.0, 200.0),
                    pred_class=0, class_scores=sc_cls0)]
    prev = [
        # prev_bad: far away (IoU ≈ 0, below gate=0.3), same class (JS=0)
        # Without pre-gating: cost = (1-0) + 1*0 = 1.0 → may beat prev_good
        # With pre-gating: blocked → Hungarian ignores it
        make_pl(box_id="prev_bad",  box=(800.0, 800.0, 900.0, 900.0),
                pred_class=0, class_scores=sc_cls0),
        # prev_good: overlapping (IoU ≈ 0.68, above gate=0.3), different class
        # cost = (1-0.68) + 1*1 = 1.32 > prev_bad's raw cost
        make_pl(box_id="prev_good", box=(110.0, 110.0, 210.0, 210.0),
                pred_class=1, class_scores=sc_cls1),
    ]

    result = matcher.match(curr, prev, image_id="img", round_t=1)
    assert len(result.matched_pairs) == 1, (
        f"curr_0 should be matched to prev_good, got: "
        f"matched={result.matched_pairs}, unmatched={result.unmatched_current}")
    assert result.matched_pairs[0] == ("curr_0", "prev_good"), (
        f"Expected curr_0 → prev_good, got {result.matched_pairs[0]}")
    print("PASS  test_gate_before_hungarian_saves_valid_match")


def test_ambiguity_only_over_valid_candidates():
    """
    Regression test for Fix 2: ambiguity gap computed over valid candidates only.

    Setup:
      curr_0 has two previous candidates:
        prev_valid:   IoU ≈ 0.68 (above gate=0.5), same class
        prev_invalid: IoU = 0.0  (below gate=0.5),  same class

    Without fix: prev_invalid contributes to gap → gap ≈ 0 → curr_0 ambiguous
    With fix: prev_invalid excluded from gap computation → only one valid
              candidate exists → cannot be ambiguous → curr_0 matched
    """
    cfg     = MatchingConfig(min_iou_gate=0.5, lambda_cls=1.0, epsilon_match=0.1)
    matcher = HungarianMatcher(cfg)

    sc = tuple([1.0] + [0.0] * 79)

    curr = [make_pl(box_id="curr_0", box=(100.0, 100.0, 200.0, 200.0),
                    class_scores=sc)]
    prev = [
        # prev_valid: IoU ≈ 0.68 (above gate=0.5), same class
        make_pl(box_id="prev_valid",
                box=(110.0, 110.0, 210.0, 210.0), class_scores=sc),
        # prev_invalid: IoU = 0.0 (below gate=0.5), same class
        # Without fix: cost(curr_0, prev_invalid) ≈ cost(curr_0, prev_valid)
        # → gap ≈ 0 → curr_0 wrongly marked ambiguous
        make_pl(box_id="prev_invalid",
                box=(800.0, 800.0, 900.0, 900.0), class_scores=sc),
    ]

    result = matcher.match(curr, prev, image_id="img", round_t=1)
    assert result.ambiguous == [], (
        f"curr_0 should NOT be ambiguous — prev_invalid is below IoU gate. "
        f"Got ambiguous={result.ambiguous}")
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0] == ("curr_0", "prev_valid")
    print("PASS  test_ambiguity_only_over_valid_candidates")


# ── MatchResult structure tests ───────────────────────────────────────────────

def test_match_result_frozen():
    result = MatchResult(
        image_id="img_0", round_t=1,
        matched_pairs=[("a", "b")],
        unmatched_current=["c"],
        ambiguous=[],
    )
    try:
        result.round_t = 2
        assert False, "MatchResult should be frozen"
    except Exception:
        pass
    print("PASS  test_match_result_frozen")


def test_matched_pairs_are_box_id_tuples():
    """matched_pairs must contain (box_id_t, box_id_prev) string tuples."""
    matcher = HungarianMatcher(MatchingConfig())
    box = (100.0, 100.0, 200.0, 200.0)
    cur  = [make_pl(box_id="cur_abc",  round_id=1, box=box)]
    prev = [make_pl(box_id="prev_xyz", round_id=0, box=box)]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert len(result.matched_pairs) == 1
    bt, bp = result.matched_pairs[0]
    assert isinstance(bt, str) and isinstance(bp, str)
    assert bt == "cur_abc"
    assert bp == "prev_xyz"
    print("PASS  test_matched_pairs_are_box_id_tuples")


def test_all_current_accounted_general():
    """
    For any result, every current box_id must appear in exactly one
    of: matched_pairs, unmatched_current, or ambiguous.
    """
    cfg     = MatchingConfig(min_iou_gate=0.3, epsilon_match=0.05)
    matcher = HungarianMatcher(cfg)

    cur = [
        make_pl(box_id="cur_0", round_id=1,
                box=(100.0, 100.0, 200.0, 200.0), pred_class=0),
        make_pl(box_id="cur_1", round_id=1,
                box=(300.0, 300.0, 400.0, 400.0), pred_class=1),
        make_pl(box_id="cur_2", round_id=1,
                box=(0.0, 0.0, 10.0, 10.0),       pred_class=2),  # no prev nearby
    ]
    prev = [
        make_pl(box_id="prev_0", round_id=0,
                box=(105.0, 105.0, 205.0, 205.0), pred_class=0),
        make_pl(box_id="prev_1", round_id=0,
                box=(310.0, 310.0, 410.0, 410.0), pred_class=1),
    ]

    result = matcher.match(cur, prev, image_id="img_0", round_t=1)
    assert all_current_accounted(result, cur)
    print("PASS  test_all_current_accounted_general")


def test_image_id_and_round_stored():
    """MatchResult must store image_id and round_t from the call."""
    matcher = HungarianMatcher(MatchingConfig())
    result  = matcher.match([], [], image_id="my_image_007", round_t=4)
    assert result.image_id == "my_image_007"
    assert result.round_t  == 4
    print("PASS  test_image_id_and_round_stored")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_empty_current,
        test_empty_previous,
        test_both_empty,
        test_perfect_match_single_pair,
        test_perfect_match_multiple_pairs,
        test_hard_gate_rejects_low_iou,
        test_hard_gate_passes_sufficient_iou,
        test_ambiguity_rejection_identical_prev_boxes,
        test_ambiguous_box_not_in_unmatched,
        test_ambiguous_prev_boxes_counted_as_removed,
        test_clear_match_not_ambiguous,
        test_lambda_cls_zero_gives_iou_only_matching,
        test_class_score_affects_matching,
        test_gate_before_hungarian_saves_valid_match,
        test_ambiguity_only_over_valid_candidates,
        test_match_result_frozen,
        test_matched_pairs_are_box_id_tuples,
        test_all_current_accounted_general,
        test_image_id_and_round_stored,
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
    print(f"Phase 3 matcher tests: {passed} passed, {failed} failed")
    print("=" * 50)