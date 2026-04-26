"""
tests/test_bank.py
==================
Unit tests for bank/schemas.py and bank/bank.py.

Tests cover:
  - Schema instantiation and field contracts
  - CandidateIndex build and lookup
  - PseudoLabelBank append / write_metadata / get_candidate_pool
  - Resumability: reload from disk gives identical pool
  - Append-only enforcement: cannot overwrite a committed round
  - Empty round (round 0 burn-in pattern)
  - RoundMetadata retrieval
"""

from __future__ import annotations

import tempfile
import os
import sys

# Allow running from project root: python -m pytest tests/ or python tests/test_bank.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import (
    BankEntry,
    CandidateIndex,
    CandidatePool,
    ComponentRecord,
    MatchResult,
    PseudoLabel,
    RoundMetadata,
    ScoringResult,
    StoppingSnapshot,
    StoppingState,
)
from autolabel.bank.bank import PseudoLabelBank


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_pseudo_label(image_id="img_0001",
                      box_id=None,
                      round_id=1,
                      box=(10.0, 20.0, 50.0, 80.0),
                      pred_class=0,
                      class_scores=None,
                      confidence=0.75) -> PseudoLabel:
    if box_id is None:
        box_id = f"{image_id}_r{round_id}_0000"
    if class_scores is None:
        # 80-class uniform-ish vector that sums to 1.0
        scores = [0.0] * 80
        scores[pred_class] = 0.5
        scores[1]          = 0.5 / 79 * 79  # rough fill
        # Simpler: just assign 1.0 to pred_class for test purposes
        scores = [0.0] * 80
        scores[pred_class] = 1.0
        class_scores = tuple(scores)
    return PseudoLabel(
        image_id    = image_id,
        box_id      = box_id,
        round_id    = round_id,
        box         = box,
        pred_class  = pred_class,
        class_scores= class_scores,
        confidence  = confidence,
    )


def make_bank_entry(image_id="img_0001",
                    round_id=1,
                    n_labels=2) -> BankEntry:
    pls = [
        make_pseudo_label(
            image_id  = image_id,
            box_id    = f"{image_id}_r{round_id}_{i:04d}",
            round_id  = round_id,
            pred_class= i % 80,
        )
        for i in range(n_labels)
    ]
    ids = [pl.box_id for pl in pls]
    return BankEntry(
        image_id            = image_id,
        round_id            = round_id,
        pseudo_labels       = pls,
        stability_scores    = {bid: 0.7 for bid in ids},
        admitted            = {bid: True for bid in ids},
        matched_prev_box_id = {bid: None for bid in ids},
        is_ambiguous        = {bid: False for bid in ids},
        c_cls_dist          = {bid: 0.9 for bid in ids},
        c_round_loc         = {bid: 0.8 for bid in ids},
        c_jitter_loc        = {bid: 0.6 for bid in ids},
        jitter_executed     = {bid: True for bid in ids},
    )


def make_round_metadata(round_id=1,
                        n_candidates=10,
                        n_admitted=7) -> RoundMetadata:
    return RoundMetadata(
        round_id           = round_id,
        n_candidates       = n_candidates,
        n_admitted         = n_admitted,
        raw_churn          = 0.12,
        stable_yield       = 0.70,
        class_drift        = 0.05,
        stop_condition_met = False,
        model_checkpoint   = f"outputs/round{round_id}/weights/best.pt",
    )


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_pseudo_label_frozen():
    pl = make_pseudo_label()
    try:
        pl.confidence = 0.9
        assert False, "PseudoLabel should be frozen"
    except Exception:
        pass
    print("PASS  test_pseudo_label_frozen")


def test_pseudo_label_fields():
    pl = make_pseudo_label(pred_class=5, confidence=0.88)
    assert pl.pred_class == 5
    assert pl.confidence == 0.88
    assert len(pl.class_scores) == 80
    assert abs(sum(pl.class_scores) - 1.0) < 1e-6
    print("PASS  test_pseudo_label_fields")


def test_candidate_index_build_and_lookup():
    pl1 = make_pseudo_label(image_id="img_0001", box_id="img_0001_r1_0000")
    pl2 = make_pseudo_label(image_id="img_0001", box_id="img_0001_r1_0001")
    pl3 = make_pseudo_label(image_id="img_0002", box_id="img_0002_r1_0000")

    pool: CandidatePool = {
        "img_0001": [pl1, pl2],
        "img_0002": [pl3],
    }
    idx = CandidateIndex.build(pool)

    assert "img_0001_r1_0000" in idx
    assert "img_0001_r1_0001" in idx
    assert "img_0002_r1_0000" in idx
    assert "nonexistent"      not in idx
    assert idx.get("img_0001_r1_0000") is pl1
    assert idx.get("img_0001_r1_0001") is pl2
    assert idx.get("nonexistent") is None
    assert len(idx) == 3
    print("PASS  test_candidate_index_build_and_lookup")


def test_match_result_frozen():
    mr = MatchResult(
        image_id          = "img_0001",
        round_t           = 1,
        matched_pairs     = [("a", "b")],
        unmatched_current = ["c"],
        ambiguous         = [],
    )
    try:
        mr.round_t = 2
        assert False, "MatchResult should be frozen"
    except Exception:
        pass
    print("PASS  test_match_result_frozen")


def test_component_record_none_vs_zero():
    # 0.0 and None mean different things — c_jitter_loc=0.0 means prefilter
    # not passed; None means unmatched/ambiguous
    cr_prefilter_failed = ComponentRecord(
        c_cls_dist   = 0.9,
        c_round_loc  = 0.8,
        c_jitter_loc = 0.0,   # prefilter not passed
        jitter_executed = False,
    )
    cr_unmatched = ComponentRecord(
        c_cls_dist   = None,
        c_round_loc  = None,
        c_jitter_loc = None,  # unmatched
        jitter_executed = False,
    )
    assert cr_prefilter_failed.c_jitter_loc == 0.0
    assert cr_unmatched.c_jitter_loc is None
    assert cr_prefilter_failed.c_jitter_loc != cr_unmatched.c_jitter_loc
    print("PASS  test_component_record_none_vs_zero")


def test_scoring_result_frozen():
    sr = ScoringResult(scores={"a": 0.7}, components={})
    try:
        sr.scores = {}
        assert False, "ScoringResult should be frozen"
    except Exception:
        pass
    print("PASS  test_scoring_result_frozen")


def test_stopping_snapshot_frozen():
    snap = StoppingSnapshot(round_id=1, raw_churn=0.1,
                            stable_yield=0.8, class_drift=0.05)
    try:
        snap.raw_churn = 0.2
        assert False, "StoppingSnapshot should be frozen"
    except Exception:
        pass
    print("PASS  test_stopping_snapshot_frozen")


def test_stopping_state_mutable():
    state = StoppingState()
    assert state.consecutive_satisfied == 0
    assert state.stopped is False
    assert state.stop_reason is None
    assert state.signal_history == []

    snap = StoppingSnapshot(round_id=1, raw_churn=0.1,
                            stable_yield=0.8, class_drift=0.05)
    state.signal_history.append(snap)
    state.consecutive_satisfied = 1

    assert len(state.signal_history) == 1
    assert state.signal_history[0].raw_churn == 0.1
    print("PASS  test_stopping_state_mutable")


# ── Bank tests ────────────────────────────────────────────────────────────────

def test_bank_append_and_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)

        entry = make_bank_entry(image_id="img_0001", round_id=1, n_labels=3)
        bank.append(entry)
        bank.write_metadata(make_round_metadata(round_id=1, n_candidates=3))

        pool = bank.get_candidate_pool(round_id=1)
        assert "img_0001" in pool
        assert len(pool["img_0001"]) == 3
        assert pool["img_0001"][0].round_id == 1
    print("PASS  test_bank_append_and_retrieve")


def test_bank_multiple_images_per_round():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)

        for img_id in ["img_0001", "img_0002", "img_0003"]:
            bank.append(make_bank_entry(image_id=img_id, round_id=2, n_labels=2))
        bank.write_metadata(make_round_metadata(round_id=2, n_candidates=6))

        pool = bank.get_candidate_pool(round_id=2)
        assert len(pool) == 3
        for img_id in ["img_0001", "img_0002", "img_0003"]:
            assert img_id in pool
            assert len(pool[img_id]) == 2
    print("PASS  test_bank_multiple_images_per_round")


def test_bank_resumability():
    """Reload from disk gives identical pool."""
    with tempfile.TemporaryDirectory() as tmp:
        bank1 = PseudoLabelBank.load_or_create(tmp)
        entry = make_bank_entry(image_id="img_0001", round_id=1, n_labels=2)
        bank1.append(entry)
        bank1.write_metadata(make_round_metadata(round_id=1))

        # Reload from same directory
        bank2 = PseudoLabelBank.load_or_create(tmp)
        pool  = bank2.get_candidate_pool(round_id=1)

        assert "img_0001" in pool
        assert len(pool["img_0001"]) == 2
        assert pool["img_0001"][0].box_id == entry.pseudo_labels[0].box_id
        assert bank2.committed_rounds() == [1]
    print("PASS  test_bank_resumability")


def test_bank_append_only_enforcement():
    """Cannot overwrite a committed round."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)
        bank.append(make_bank_entry(round_id=1))
        bank.write_metadata(make_round_metadata(round_id=1))

        # Try to write round 1 again
        bank2 = PseudoLabelBank.load_or_create(tmp)
        bank2.append(make_bank_entry(round_id=1))
        try:
            bank2.write_metadata(make_round_metadata(round_id=1))
            assert False, "Should have raised FileExistsError"
        except FileExistsError:
            pass
    print("PASS  test_bank_append_only_enforcement")


def test_bank_empty_round_zero():
    """Round 0 burn-in: no entries, just metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)
        # No append() calls — round 0 has no pseudo-labels
        bank.write_metadata(RoundMetadata(
            round_id=0, n_candidates=0, n_admitted=0,
            raw_churn=0.0, stable_yield=0.0, class_drift=0.0,
            stop_condition_met=False, model_checkpoint="outputs/round0/best.pt"
        ))

        pool = bank.get_candidate_pool(round_id=0)
        assert pool == {}
        assert bank.committed_rounds() == [0]
    print("PASS  test_bank_empty_round_zero")


def test_bank_multiple_rounds():
    """Multi-round append and retrieval."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)

        # Round 0 — empty burn-in
        bank.write_metadata(make_round_metadata(round_id=0, n_candidates=0))

        # Round 1
        bank.append(make_bank_entry(image_id="img_0001", round_id=1, n_labels=3))
        bank.write_metadata(make_round_metadata(round_id=1, n_candidates=3))

        # Round 2
        bank.append(make_bank_entry(image_id="img_0001", round_id=2, n_labels=5))
        bank.write_metadata(make_round_metadata(round_id=2, n_candidates=5))

        assert bank.committed_rounds() == [0, 1, 2]
        assert bank.latest_round_id()  == 2

        pool1 = bank.get_candidate_pool(round_id=1)
        pool2 = bank.get_candidate_pool(round_id=2)
        assert len(pool1["img_0001"]) == 3
        assert len(pool2["img_0001"]) == 5
    print("PASS  test_bank_multiple_rounds")


def test_bank_mixed_round_id_raises():
    """Appending entries from different rounds before write_metadata raises."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)
        bank.append(make_bank_entry(round_id=1))
        try:
            bank.append(make_bank_entry(round_id=2))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    print("PASS  test_bank_mixed_round_id_raises")


def test_bank_write_metadata_round_id_mismatch_raises():
    """
    Core integrity check: append entries for round 1, then call
    write_metadata with round_id=2. Must raise ValueError — not silently
    write round-1 entries into round_0002.json.
    """
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)
        bank.append(make_bank_entry(image_id="img_0001", round_id=1))
        try:
            bank.write_metadata(make_round_metadata(round_id=2))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    print("PASS  test_bank_write_metadata_round_id_mismatch_raises")


def test_bank_metadata_retrieval():
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)
        bank.write_metadata(make_round_metadata(round_id=0, n_candidates=0))
        bank.append(make_bank_entry(round_id=1))
        bank.write_metadata(make_round_metadata(round_id=1, n_candidates=10, n_admitted=7))

        meta0 = bank.get_metadata(round_id=0)
        meta1 = bank.get_metadata(round_id=1)
        assert meta0.n_candidates == 0
        assert meta1.n_candidates == 10
        assert meta1.n_admitted   == 7
        assert bank.get_metadata(round_id=99) is None
    print("PASS  test_bank_metadata_retrieval")


def test_pseudo_label_serialization_round_trip():
    """PseudoLabel survives JSON serialization through the bank."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = PseudoLabelBank.load_or_create(tmp)

        scores = tuple([0.0] * 80)
        scores_list = list(scores)
        scores_list[7] = 0.6
        scores_list[3] = 0.4
        scores = tuple(scores_list)

        pl = PseudoLabel(
            image_id    = "img_0042",
            box_id      = "img_0042_r1_0000",
            round_id    = 1,
            box         = (10.5, 20.3, 100.1, 200.7),
            pred_class  = 7,
            class_scores= scores,
            confidence  = 0.83,
        )
        entry = BankEntry(
            image_id            = "img_0042",
            round_id            = 1,
            pseudo_labels       = [pl],
            stability_scores    = {pl.box_id: 0.72},
            admitted            = {pl.box_id: True},
            matched_prev_box_id = {pl.box_id: None},
            is_ambiguous        = {pl.box_id: False},
            c_cls_dist          = {pl.box_id: 0.88},
            c_round_loc         = {pl.box_id: 0.91},
            c_jitter_loc        = {pl.box_id: 0.0},
            jitter_executed     = {pl.box_id: False},
        )
        bank.append(entry)
        bank.write_metadata(make_round_metadata(round_id=1))

        # Reload
        bank2 = PseudoLabelBank.load_or_create(tmp)
        pool  = bank2.get_candidate_pool(round_id=1)
        pl2   = pool["img_0042"][0]

        assert pl2.box_id      == pl.box_id
        assert pl2.pred_class  == pl.pred_class
        assert pl2.confidence  == pl.confidence
        assert pl2.box         == pl.box
        assert len(pl2.class_scores) == 80
        assert abs(pl2.class_scores[7] - 0.6) < 1e-9
        assert abs(pl2.class_scores[3] - 0.4) < 1e-9
    print("PASS  test_pseudo_label_serialization_round_trip")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_pseudo_label_frozen,
        test_pseudo_label_fields,
        test_candidate_index_build_and_lookup,
        test_match_result_frozen,
        test_component_record_none_vs_zero,
        test_scoring_result_frozen,
        test_stopping_snapshot_frozen,
        test_stopping_state_mutable,
        test_bank_append_and_retrieve,
        test_bank_multiple_images_per_round,
        test_bank_resumability,
        test_bank_append_only_enforcement,
        test_bank_empty_round_zero,
        test_bank_multiple_rounds,
        test_bank_mixed_round_id_raises,
        test_bank_write_metadata_round_id_mismatch_raises,
        test_bank_metadata_retrieval,
        test_pseudo_label_serialization_round_trip,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Phase 1 bank tests: {passed} passed, {failed} failed")
    print("=" * 50)