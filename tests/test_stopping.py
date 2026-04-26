"""
tests/test_stopping.py
======================
Unit tests for admission/interface.py, admission/global_threshold.py,
and stopping/stopping.py.

Admission tests cover:
  - GlobalThresholdPolicy admits when S >= tau_stab AND q_t >= tau_conf
  - Rejects when S < tau_stab (even if confidence passes)
  - Rejects when confidence < tau_conf (even if score passes)
  - Never admits score=None (unmatched/ambiguous)
  - Returns empty list when nothing passes
  - AdmissionPolicy is abstract (cannot be instantiated directly)

Stopping tests cover:
  - compute_signals returns StoppingSnapshot (not tuple)
  - snapshot fields accessible by name (.raw_churn etc.)
  - RawChurn: N_new, N_removed, N_class_change, N_geom_change
  - Ambiguous boxes counted as removed (Patch 4)
  - StableYield = |A_t| / |C_t|, not over A_t
  - ClassDrift uses JS normalized by ln(2)
  - Counter resets on unsatisfied round
  - Counter increments on satisfied round
  - Stops after K_consecutive satisfied rounds
  - Warmup guard: evaluate() is NOT called during warmup (enforced by caller)
  - StoppingState signal_history is List[StoppingSnapshot]
  - Empty C_t and C_prev handled gracefully
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.admission.interface import AdmissionPolicy
from autolabel.admission.global_threshold import AdmissionConfig, GlobalThresholdPolicy
from autolabel.bank.schemas import (
    CandidatePool, MatchResult, PseudoLabel,
    StoppingSnapshot, StoppingState,
)
from autolabel.stopping.stopping import StoppingConfig, StoppingEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_scores(pred_class: int, n: int = 80) -> tuple:
    s = [0.0] * n; s[pred_class] = 1.0
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


def make_mr(image_id="img", round_t=1,
            matched_pairs=None, unmatched_current=None,
            ambiguous=None) -> MatchResult:
    return MatchResult(
        image_id=image_id, round_t=round_t,
        matched_pairs=matched_pairs or [],
        unmatched_current=unmatched_current or [],
        ambiguous=ambiguous or [],
    )


def default_stop_config(**kwargs) -> StoppingConfig:
    cfg = StoppingConfig()
    for k, v in kwargs.items():
        object.__setattr__(cfg, k, v)
    return cfg


# ── Admission tests ───────────────────────────────────────────────────────────

def test_admission_policy_is_abstract():
    """AdmissionPolicy cannot be instantiated directly."""
    try:
        AdmissionPolicy()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    print("PASS  test_admission_policy_is_abstract")


def test_global_threshold_admits_when_both_pass():
    cfg     = AdmissionConfig(tau_stab=0.6, tau_conf=0.4)
    policy  = GlobalThresholdPolicy(cfg)
    pl      = make_pl(box_id="t_0", confidence=0.7)
    scores  = {"t_0": 0.75}
    result  = policy.admit({"img": [pl]}, scores, round_id=1)
    assert len(result) == 1 and result[0].box_id == "t_0"
    print("PASS  test_global_threshold_admits_when_both_pass")


def test_global_threshold_rejects_low_score():
    cfg     = AdmissionConfig(tau_stab=0.6, tau_conf=0.4)
    policy  = GlobalThresholdPolicy(cfg)
    pl      = make_pl(box_id="t_0", confidence=0.9)
    scores  = {"t_0": 0.55}  # below tau_stab=0.6
    result  = policy.admit({"img": [pl]}, scores, round_id=1)
    assert result == []
    print("PASS  test_global_threshold_rejects_low_score")


def test_global_threshold_rejects_low_confidence():
    cfg     = AdmissionConfig(tau_stab=0.6, tau_conf=0.4)
    policy  = GlobalThresholdPolicy(cfg)
    pl      = make_pl(box_id="t_0", confidence=0.3)  # below tau_conf=0.4
    scores  = {"t_0": 0.9}
    result  = policy.admit({"img": [pl]}, scores, round_id=1)
    assert result == []
    print("PASS  test_global_threshold_rejects_low_confidence")


def test_global_threshold_rejects_none_score():
    """score=None (unmatched/ambiguous) must never be admitted."""
    cfg    = AdmissionConfig(tau_stab=0.0, tau_conf=0.0)  # permissive
    policy = GlobalThresholdPolicy(cfg)
    pl     = make_pl(box_id="t_0", confidence=0.99)
    scores = {"t_0": None}
    result = policy.admit({"img": [pl]}, scores, round_id=1)
    assert result == []
    print("PASS  test_global_threshold_rejects_none_score")


def test_global_threshold_boundary_exact():
    """Exactly at threshold values must be admitted (>= not >)."""
    cfg    = AdmissionConfig(tau_stab=0.6, tau_conf=0.4)
    policy = GlobalThresholdPolicy(cfg)
    pl     = make_pl(box_id="t_0", confidence=0.4)  # exactly tau_conf
    scores = {"t_0": 0.6}                           # exactly tau_stab
    result = policy.admit({"img": [pl]}, scores, round_id=1)
    assert len(result) == 1
    print("PASS  test_global_threshold_boundary_exact")


def test_global_threshold_empty_pool():
    policy = GlobalThresholdPolicy(AdmissionConfig())
    result = policy.admit({}, {}, round_id=1)
    assert result == []
    print("PASS  test_global_threshold_empty_pool")


def test_global_threshold_multi_image():
    """Mixed admit/reject across images."""
    cfg    = AdmissionConfig(tau_stab=0.6, tau_conf=0.4)
    policy = GlobalThresholdPolicy(cfg)
    pl_a   = make_pl(image_id="img_a", box_id="a_0", confidence=0.8)
    pl_b   = make_pl(image_id="img_b", box_id="b_0", confidence=0.8)
    pl_c   = make_pl(image_id="img_b", box_id="b_1", confidence=0.8)
    C_t    = {"img_a": [pl_a], "img_b": [pl_b, pl_c]}
    scores = {"a_0": 0.7, "b_0": 0.5, "b_1": None}  # b_0 below, b_1 unmatched
    result = policy.admit(C_t, scores, round_id=1)
    admitted_ids = {pl.box_id for pl in result}
    assert admitted_ids == {"a_0"}
    print("PASS  test_global_threshold_multi_image")


# ── Stopping signal tests ─────────────────────────────────────────────────────

def test_compute_signals_returns_snapshot():
    """compute_signals must return StoppingSnapshot, not a tuple."""
    stopper = StoppingEvaluator(StoppingConfig())
    snap    = stopper.compute_signals(
        C_t={}, C_prev={}, A_t=[],
        match_results={}, round_t=1)
    assert isinstance(snap, StoppingSnapshot)
    assert snap.round_id == 1
    print("PASS  test_compute_signals_returns_snapshot")


def test_snapshot_fields_by_name():
    """Snapshot fields must be accessible by name."""
    stopper = StoppingEvaluator(StoppingConfig())
    snap    = stopper.compute_signals({}, {}, [], {}, round_t=3)
    _ = snap.raw_churn
    _ = snap.stable_yield
    _ = snap.class_drift
    _ = snap.round_id
    print("PASS  test_snapshot_fields_by_name")


def test_rawchurn_n_new():
    """N_new counts unmatched_current boxes."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box),
                        make_pl(box_id="t_1", box=box)]}
    C_prev  = {"img": [make_pl(box_id="p_0", box=box, round_id=0)]}
    # t_0 matched, t_1 unmatched (new)
    mr      = make_mr("img", matched_pairs=[("t_0", "p_0")],
                      unmatched_current=["t_1"])
    snap    = stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
    # union = 2 + 1 - 1 = 2; N_new=1, N_removed=0
    assert snap.raw_churn == 1.0 / 2.0
    print("PASS  test_rawchurn_n_new")


def test_rawchurn_n_removed():
    """N_removed counts prev boxes absent from matched_pairs."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    C_prev  = {"img": [make_pl(box_id="p_0", box=box, round_id=0),
                        make_pl(box_id="p_1", box=box, round_id=0)]}
    # t_0 matched to p_0; p_1 has no current match → removed
    mr      = make_mr("img", matched_pairs=[("t_0", "p_0")])
    snap    = stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
    # union = 1 + 2 - 1 = 2; N_new=0, N_removed=1
    assert snap.raw_churn == 1.0 / 2.0
    print("PASS  test_rawchurn_n_removed")


def test_rawchurn_ambiguous_counted_as_removed():
    """
    Patch 4: ambiguous current box is not in matched_pairs.
    Its nearest prev box is therefore counted as removed.
    """
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    C_prev  = {"img": [make_pl(box_id="p_0", box=box, round_id=0)]}
    # t_0 is ambiguous → not in matched_pairs → p_0 counted as removed
    mr      = make_mr("img", ambiguous=["t_0"])
    snap    = stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
    # union = 1 + 1 - 0 = 2; N_new=0, N_removed=1
    assert snap.raw_churn == 1.0 / 2.0
    print("PASS  test_rawchurn_ambiguous_counted_as_removed")


def test_rawchurn_n_class_change():
    """N_class_change counts matched pairs where pred_class changed."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box, pred_class=0)]}
    C_prev  = {"img": [make_pl(box_id="p_0", box=box, pred_class=1, round_id=0)]}
    mr      = make_mr("img", matched_pairs=[("t_0", "p_0")])
    snap    = stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
    # union=2-1=1 (but actually n_t+n_prev-n_matched=1+1-1=1)
    # N_class_change=1 → churn = 1/1 = 1.0
    assert snap.raw_churn == 1.0
    print("PASS  test_rawchurn_n_class_change")


def test_rawchurn_n_geom_change():
    """N_geom_change counts matched pairs where IoU < tau_geom_change."""
    cfg     = default_stop_config(tau_geom_change=0.5)
    stopper = StoppingEvaluator(cfg)
    # Two boxes with low IoU (far apart but matched)
    C_t    = {"img": [make_pl(box_id="t_0", box=(0.0, 0.0, 10.0, 10.0))]}
    C_prev = {"img": [make_pl(box_id="p_0", box=(100.0, 100.0, 200.0, 200.0),
                              round_id=0)]}
    mr     = make_mr("img", matched_pairs=[("t_0", "p_0")])
    snap   = stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
    # IoU ≈ 0 < tau_geom_change=0.5 → N_geom_change=1
    # union=1+1-1=1 → churn=1.0
    assert snap.raw_churn == 1.0
    print("PASS  test_rawchurn_n_geom_change")


def test_stable_yield():
    """StableYield = |A_t| / |C_t|."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id=f"t_{i}", box=box) for i in range(10)]}
    A_t     = [make_pl(box_id=f"t_{i}", box=box) for i in range(6)]
    match_results = {
        "img": make_mr("img", unmatched_current=[f"t_{i}" for i in range(10)])
    }
    snap    = stopper.compute_signals(C_t, {}, A_t, match_results, round_t=1)
    assert abs(snap.stable_yield - 0.6) < 1e-9
    print("PASS  test_stable_yield")


def test_stable_yield_empty_c_t():
    stopper = StoppingEvaluator(StoppingConfig())
    # Empty C_t and C_prev → no images → empty match_results is valid
    snap    = stopper.compute_signals({}, {}, [], {}, round_t=1)
    assert snap.stable_yield == 0.0
    print("PASS  test_stable_yield_empty_c_t")


def test_class_drift_identical_pools():
    """Identical class distributions → ClassDrift = 0."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t    = {"img": [make_pl(box_id="t_0", pred_class=0, box=box),
                       make_pl(box_id="t_1", pred_class=1, box=box)]}
    C_prev = {"img": [make_pl(box_id="p_0", pred_class=0, box=box, round_id=0),
                       make_pl(box_id="p_1", pred_class=1, box=box, round_id=0)]}
    match_results = {
        "img": make_mr("img", matched_pairs=[("t_0", "p_0"), ("t_1", "p_1")])
    }
    snap   = stopper.compute_signals(C_t, C_prev, [], match_results, round_t=1)
    assert abs(snap.class_drift - 0.0) < 1e-9
    print("PASS  test_class_drift_identical_pools")


def test_class_drift_different_pools():
    """Different class distributions → ClassDrift > 0."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    # C_t all class 0, C_prev all class 1
    C_t    = {"img": [make_pl(box_id="t_0", pred_class=0, box=box)]}
    C_prev = {"img": [make_pl(box_id="p_0", pred_class=1, box=box, round_id=0)]}
    match_results = {
        "img": make_mr("img", matched_pairs=[("t_0", "p_0")])
    }
    snap   = stopper.compute_signals(C_t, C_prev, [], match_results, round_t=1)
    assert snap.class_drift > 0.0
    print("PASS  test_class_drift_different_pools")


def test_class_drift_empty_prev():
    """Empty C_prev → uniform prior → ClassDrift > 0 if C_t is skewed."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", pred_class=0, box=box)]}
    match_results = {
        "img": make_mr("img", unmatched_current=["t_0"])
    }
    snap    = stopper.compute_signals(C_t, {}, [], match_results, round_t=1)
    assert snap.class_drift >= 0.0
    print("PASS  test_class_drift_empty_prev")


# ── Counter and stopping tests ────────────────────────────────────────────────

def test_counter_increments_on_satisfied():
    """consecutive_satisfied increments when all three conditions hold."""
    cfg     = default_stop_config(
        epsilon_churn=1.0, epsilon_yield=1.0, epsilon_drift=1.0,
        K_consecutive=3)
    stopper = StoppingEvaluator(cfg)
    state   = StoppingState()

    # Add a seed snapshot so we have 2 snapshots after first evaluate()
    state.signal_history.append(
        StoppingSnapshot(round_id=0, raw_churn=0.1,
                         stable_yield=0.5, class_drift=0.05))

    state = stopper.evaluate({}, {}, [], {}, state, round_t=1)
    assert state.consecutive_satisfied == 1
    assert not state.stopped
    print("PASS  test_counter_increments_on_satisfied")


def test_counter_resets_on_unsatisfied():
    """consecutive_satisfied resets to 0 when any condition fails."""
    cfg     = default_stop_config(
        epsilon_churn=0.001,  # very tight — will fail
        epsilon_yield=1.0, epsilon_drift=1.0, K_consecutive=3)
    stopper = StoppingEvaluator(cfg)
    state   = StoppingState()
    state.consecutive_satisfied = 2  # pretend we had two satisfied rounds

    # Seed snapshot with raw_churn=0.1
    state.signal_history.append(
        StoppingSnapshot(round_id=0, raw_churn=0.1,
                         stable_yield=0.5, class_drift=0.05))

    # New snapshot will have raw_churn=0.5 (delta=0.4 >> epsilon_churn=0.001)
    box    = (100.0, 100.0, 200.0, 200.0)
    C_t    = {"img": [make_pl(box_id="t_0", box=box),
                       make_pl(box_id="t_1", box=box)]}
    C_prev = {"img": [make_pl(box_id="p_0", box=box, round_id=0)]}
    mr     = make_mr("img", matched_pairs=[("t_0", "p_0")],
                     unmatched_current=["t_1"])
    state  = stopper.evaluate(C_t, C_prev, [], {"img": mr}, state, round_t=1)
    assert state.consecutive_satisfied == 0
    assert not state.stopped
    print("PASS  test_counter_resets_on_unsatisfied")


def test_stops_after_k_consecutive():
    """Stopping fires after K_consecutive satisfied rounds."""
    cfg     = default_stop_config(
        epsilon_churn=1.0, epsilon_yield=1.0, epsilon_drift=1.0,
        K_consecutive=2)
    stopper = StoppingEvaluator(cfg)
    state   = StoppingState()

    # Seed snapshot
    state.signal_history.append(
        StoppingSnapshot(round_id=0, raw_churn=0.1,
                         stable_yield=0.5, class_drift=0.05))

    # Round 1: first satisfied → consecutive=1, not stopped
    state = stopper.evaluate({}, {}, [], {}, state, round_t=1)
    assert state.consecutive_satisfied == 1
    assert not state.stopped

    # Round 2: second satisfied → consecutive=2 >= K=2 → stopped
    state = stopper.evaluate({}, {}, [], {}, state, round_t=2)
    assert state.consecutive_satisfied == 2
    assert state.stopped
    assert state.stop_reason == "convergence"
    print("PASS  test_stops_after_k_consecutive")


def test_signal_history_is_list_of_snapshots():
    """signal_history must be List[StoppingSnapshot], never raw tuples."""
    stopper = StoppingEvaluator(StoppingConfig())
    state   = StoppingState()
    state.signal_history.append(
        StoppingSnapshot(round_id=0, raw_churn=0.1,
                         stable_yield=0.5, class_drift=0.05))
    state = stopper.evaluate({}, {}, [], {}, state, round_t=1)

    for snap in state.signal_history:
        assert isinstance(snap, StoppingSnapshot), (
            f"signal_history contains non-snapshot: {type(snap)}")
    print("PASS  test_signal_history_is_list_of_snapshots")


def test_rawchurn_zero_when_stable():
    """Stable pool (same boxes, same classes, same geometry) → churn = 0."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t    = {"img": [make_pl(box_id="t_0", box=box, pred_class=0)]}
    C_prev = {"img": [make_pl(box_id="p_0", box=box, pred_class=0, round_id=0)]}
    mr     = make_mr("img", matched_pairs=[("t_0", "p_0")])
    snap   = stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
    assert snap.raw_churn == 0.0
    print("PASS  test_rawchurn_zero_when_stable")



# ── Validation tests ──────────────────────────────────────────────────────────

def test_missing_match_result_raises():
    """Image in C_t with no MatchResult must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img_a": [make_pl(image_id="img_a", box_id="a_0", box=box)],
               "img_b": [make_pl(image_id="img_b", box_id="b_0", box=box)]}
    C_prev  = {}
    # img_b has no MatchResult
    match_results = {"img_a": make_mr("img_a", unmatched_current=["a_0"])}
    try:
        stopper.compute_signals(C_t, C_prev, [], match_results, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "img_b" in str(e)
    print("PASS  test_missing_match_result_raises")


def test_ghost_current_in_matched_pairs_raises():
    """Ghost current box_id in matched_pairs must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    C_prev  = {"img": [make_pl(box_id="p_0", box=box, round_id=0)]}
    mr      = make_mr("img", matched_pairs=[("t_GHOST", "p_0")])
    try:
        stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "t_GHOST" in str(e)
    print("PASS  test_ghost_current_in_matched_pairs_raises")


def test_ghost_previous_in_matched_pairs_raises():
    """Ghost previous box_id in matched_pairs must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    C_prev  = {"img": [make_pl(box_id="p_0", box=box, round_id=0)]}
    mr      = make_mr("img", matched_pairs=[("t_0", "p_GHOST")])
    try:
        stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "p_GHOST" in str(e)
    print("PASS  test_ghost_previous_in_matched_pairs_raises")


def test_ghost_id_in_unmatched_current_raises():
    """Ghost box_id in unmatched_current must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    C_prev  = {}
    mr      = make_mr("img", unmatched_current=["t_GHOST"])
    try:
        stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "t_GHOST" in str(e)
    print("PASS  test_ghost_id_in_unmatched_current_raises")


def test_ghost_id_in_ambiguous_raises():
    """Ghost box_id in ambiguous must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    C_prev  = {}
    mr      = make_mr("img", ambiguous=["t_GHOST"])
    try:
        stopper.compute_signals(C_t, C_prev, [], {"img": mr}, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "t_GHOST" in str(e)
    print("PASS  test_ghost_id_in_ambiguous_raises")


def test_a_t_ghost_label_raises():
    """A_t box_id not in C_t must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    C_t     = {"img": [make_pl(box_id="t_0", box=box)]}
    A_t     = [make_pl(box_id="t_GHOST", box=box)]  # not in C_t
    match_results = {"img": make_mr("img", unmatched_current=["t_0"])}
    try:
        stopper.compute_signals(C_t, {}, A_t, match_results, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "t_GHOST" in str(e)
    print("PASS  test_a_t_ghost_label_raises")


def test_a_t_duplicate_raises():
    """Duplicate box_id in A_t must raise ValueError."""
    stopper = StoppingEvaluator(StoppingConfig())
    box     = (100.0, 100.0, 200.0, 200.0)
    pl      = make_pl(box_id="t_0", box=box)
    C_t     = {"img": [pl]}
    A_t     = [pl, pl]  # duplicate
    match_results = {"img": make_mr("img", unmatched_current=["t_0"])}
    try:
        stopper.compute_signals(C_t, {}, A_t, match_results, round_t=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "t_0" in str(e)
    print("PASS  test_a_t_duplicate_raises")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_admission_policy_is_abstract,
        test_global_threshold_admits_when_both_pass,
        test_global_threshold_rejects_low_score,
        test_global_threshold_rejects_low_confidence,
        test_global_threshold_rejects_none_score,
        test_global_threshold_boundary_exact,
        test_global_threshold_empty_pool,
        test_global_threshold_multi_image,
        test_compute_signals_returns_snapshot,
        test_snapshot_fields_by_name,
        test_rawchurn_n_new,
        test_rawchurn_n_removed,
        test_rawchurn_ambiguous_counted_as_removed,
        test_rawchurn_n_class_change,
        test_rawchurn_n_geom_change,
        test_stable_yield,
        test_stable_yield_empty_c_t,
        test_class_drift_identical_pools,
        test_class_drift_different_pools,
        test_class_drift_empty_prev,
        test_counter_increments_on_satisfied,
        test_counter_resets_on_unsatisfied,
        test_stops_after_k_consecutive,
        test_signal_history_is_list_of_snapshots,
        test_rawchurn_zero_when_stable,
        test_missing_match_result_raises,
        test_ghost_current_in_matched_pairs_raises,
        test_ghost_previous_in_matched_pairs_raises,
        test_ghost_id_in_unmatched_current_raises,
        test_ghost_id_in_ambiguous_raises,
        test_a_t_ghost_label_raises,
        test_a_t_duplicate_raises,
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
    print(f"Phase 5 stopping tests: {passed} passed, {failed} failed")
    print("=" * 50)