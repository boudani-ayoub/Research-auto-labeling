"""
tests/test_round_runner_smoke.py
================================
Phase 6A smoke test: orchestrator control loop with synthetic data.

No YOLO training. No real images. No GPU.

Uses:
  - Synthetic CandidatePools (pre-built PseudoLabel lists)
  - Stub trainer that returns a fake checkpoint path
  - Stub infer_fn that returns pre-built pools round by round
  - Pre-computed jitter_ious (None — jitter_infer_fn not set)

Verifies:
  - 2-round loop runs end-to-end without errors
  - All bank fields populated for every image every round
  - RoundMetadata written for every round including round 0
  - StoppingState signal_history populated
  - StoppingSnapshot fields accessible by name
  - Admitted set A_t is a subset of C_t
  - Bank is resumable (reload after 2 rounds)
  - warmup guard: stopper.evaluate() not called during warmup
  - max_rounds cap fires correctly
"""

from __future__ import annotations

import tempfile
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autolabel.bank.schemas import (
    CandidatePool, PseudoLabel, StoppingSnapshot,
)
from autolabel.bank.bank import PseudoLabelBank
from autolabel.orchestrator import (
    Orchestrator, PipelineConfig, TrainerInterface,
)
from autolabel.matching.matcher import MatchingConfig
from autolabel.scoring.scorer import ScoringConfig
from autolabel.admission.global_threshold import AdmissionConfig
from autolabel.stopping.stopping import StoppingConfig


# ── Synthetic data builders ───────────────────────────────────────────────────

def make_scores(pred_class: int, nc: int = 80) -> tuple:
    s = [0.0] * nc
    s[pred_class] = 1.0
    return tuple(s)


def make_pl(image_id, box_id, round_id, box, pred_class=0,
            confidence=0.8) -> PseudoLabel:
    return PseudoLabel(
        image_id    = image_id,
        box_id      = box_id,
        round_id    = round_id,
        box         = box,
        pred_class  = pred_class,
        class_scores= make_scores(pred_class),
        confidence  = confidence,
    )


def build_synthetic_pools(n_images: int = 3,
                           n_boxes: int  = 2,
                           n_rounds: int = 2
                           ) -> List[CandidatePool]:
    """
    Build n_rounds CandidatePools with slightly drifting box positions.
    Each pool has n_images images, each with n_boxes detections.
    """
    pools = []
    for r in range(1, n_rounds + 1):
        pool: CandidatePool = {}
        for img_i in range(n_images):
            image_id = f"img_{img_i:04d}"
            pls = []
            for box_i in range(n_boxes):
                # Small drift each round so boxes still match across rounds
                offset = float(r * 2)
                x1 = 100.0 + box_i * 200 + offset
                y1 = 100.0 + box_i * 150 + offset
                x2 = x1 + 80.0
                y2 = y1 + 80.0
                pls.append(make_pl(
                    image_id  = image_id,
                    box_id    = f"{image_id}_r{r}_{box_i:04d}",
                    round_id  = r,
                    box       = (x1, y1, x2, y2),
                    pred_class= box_i % 80,
                    confidence= 0.75,
                ))
            pool[image_id] = pls
        pools.append(pool)
    return pools


# ── Stubs ─────────────────────────────────────────────────────────────────────

class StubTrainer(TrainerInterface):
    """Returns fake checkpoint paths without running any training."""
    def train(self, labeled_data, pseudo_labels, round_id):
        return f"outputs/stub_round{round_id}/weights/best.pt"


def make_stub_infer_fn(pools: List[CandidatePool]):
    """
    Returns an infer_fn that serves pre-built pools in order.
    Round 1 → pools[0], Round 2 → pools[1], etc.
    """
    call_count = [0]
    def infer_fn(model, image_ids, images, round_id):
        idx = min(round_id - 1, len(pools) - 1)
        return pools[idx]
    return infer_fn


def make_stub_images(n_images: int = 3
                      ) -> Tuple[List[str], List[np.ndarray]]:
    """Returns (image_ids, images) for the unlabeled set."""
    image_ids = [f"img_{i:04d}" for i in range(n_images)]
    images    = [np.zeros((640, 640, 3), dtype=np.uint8)
                 for _ in range(n_images)]
    return image_ids, images


def make_stub_load_model():
    """Returns a callable that returns a fake model object."""
    return lambda checkpoint: f"model_from_{checkpoint}"


# ── Smoke tests ───────────────────────────────────────────────────────────────

def make_config(bank_path: str,
                max_rounds:    int   = 3,
                warmup_rounds: int   = 1,
                tau_stab:      float = 0.0,
                tau_conf:      float = 0.0) -> PipelineConfig:
    """
    Permissive config: low thresholds so most boxes are admitted,
    loose stopping so pipeline runs to max_rounds.
    tau_stab=0, tau_conf=0 admits everything with a non-None score.
    """
    return PipelineConfig(
        max_rounds    = max_rounds,
        warmup_rounds = warmup_rounds,
        bank_path     = bank_path,
        matching      = MatchingConfig(min_iou_gate=0.1, epsilon_match=0.001),
        scoring       = ScoringConfig(alpha=0.33, beta=0.33, gamma=0.34,
                                      tau_pre=0.1),
        admission     = AdmissionConfig(tau_stab=tau_stab, tau_conf=tau_conf),
        stopping      = StoppingConfig(
            epsilon_churn=1.0, epsilon_yield=1.0, epsilon_drift=1.0,
            K_consecutive=99,  # don't stop early
        ),
    )


def test_two_round_loop_runs():
    """End-to-end: 2 rounds complete without raising."""
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 3
        pools    = build_synthetic_pools(n_images=n_images, n_boxes=2, n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        checkpoint, bank, stop_state = orc.run(labeled_data="fake_dataset")
        assert checkpoint is not None
        assert stop_state.stopped
        assert stop_state.stop_reason == "max_rounds"
    print("PASS  test_two_round_loop_runs")


def test_round_metadata_written_for_all_rounds():
    """RoundMetadata must exist for round 0 and all subsequent rounds."""
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 2
        pools    = build_synthetic_pools(n_images=n_images, n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        _, bank, _ = orc.run("fake")

        rounds = bank.committed_rounds()
        assert 0 in rounds, "Round 0 metadata missing"
        assert 1 in rounds, "Round 1 metadata missing"
        assert 2 in rounds, "Round 2 metadata missing"
    print("PASS  test_round_metadata_written_for_all_rounds")


def test_bank_entries_fully_populated():
    """Every box in C_t must have all BankEntry fields populated in the bank."""
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        n_images, n_boxes = 2, 3
        pools    = build_synthetic_pools(n_images=n_images, n_boxes=n_boxes,
                                          n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1,
                          tau_stab=0.0, tau_conf=0.0)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        _, bank, _ = orc.run("fake")

        # Read raw round file and check all BankEntry fields are present
        # for every box in every image
        required_fields = [
            "stability_scores", "admitted", "matched_prev_box_id",
            "is_ambiguous", "c_cls_dist", "c_round_loc",
            "c_jitter_loc", "jitter_executed",
        ]

        for round_id in [1, 2]:
            round_path = Path(tmp) / f"round_{round_id:04d}.json"
            assert round_path.exists(), f"round_{round_id:04d}.json missing"
            with open(round_path) as f:
                entries = json.load(f)

            assert len(entries) == n_images, (
                f"Round {round_id}: expected {n_images} entries, "
                f"got {len(entries)}")

            for entry in entries:
                pls = entry["pseudo_labels"]
                assert len(pls) == n_boxes, (
                    f"Round {round_id} image {entry['image_id']}: "
                    f"expected {n_boxes} boxes")

                for field in required_fields:
                    assert field in entry, (
                        f"Round {round_id} entry missing field: {field!r}")
                    assert isinstance(entry[field], dict), (
                        f"Round {round_id} field {field!r} is not a dict")

                    for pl in pls:
                        assert pl["box_id"] in entry[field], (
                            f"Round {round_id} box {pl['box_id']!r} "
                            f"missing from field {field!r}")

    print("PASS  test_bank_entries_fully_populated")


def test_admitted_is_subset_of_c_t():
    """n_admitted <= n_candidates for every non-warmup round."""
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 2
        pools    = build_synthetic_pools(n_images=n_images, n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1,
                          tau_stab=0.0, tau_conf=0.0)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        _, bank, _ = orc.run("fake")

        for meta in bank.all_metadata():
            if meta.round_id == 0:
                continue
            assert meta.n_admitted <= meta.n_candidates, (
                f"Round {meta.round_id}: n_admitted={meta.n_admitted} "
                f"> n_candidates={meta.n_candidates}")
    print("PASS  test_admitted_is_subset_of_c_t")


def test_signal_history_populated():
    """signal_history must have a snapshot for every non-warmup round."""
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 2
        pools    = build_synthetic_pools(n_images=n_images, n_rounds=3)
        img_ids, imgs = make_stub_images(n_images)

        # warmup_rounds=1 → rounds 1 is warmup, 2 and 3 call evaluate()
        cfg = make_config(tmp, max_rounds=3, warmup_rounds=1)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        _, _, stop_state = orc.run("fake")

        # 3 rounds → 3 snapshots in signal_history
        assert len(stop_state.signal_history) == 3
        for snap in stop_state.signal_history:
            assert isinstance(snap, StoppingSnapshot)
            # Access by name — never by index
            _ = snap.raw_churn
            _ = snap.stable_yield
            _ = snap.class_drift
    print("PASS  test_signal_history_populated")


def test_warmup_guard():
    """
    During warmup rounds, stopper.evaluate() must NOT be called.
    Only compute_signals() appends to signal_history.
    We verify by checking consecutive_satisfied stays 0 during warmup.
    """
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 2
        pools    = build_synthetic_pools(n_images=n_images, n_rounds=1)
        img_ids, imgs = make_stub_images(n_images)

        # warmup_rounds=2 → only round 1 runs (within warmup)
        cfg = make_config(tmp, max_rounds=1, warmup_rounds=2)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        _, _, stop_state = orc.run("fake")

        # Warmup only → evaluate() never called → consecutive_satisfied = 0
        assert stop_state.consecutive_satisfied == 0
        # But signal_history still populated via compute_signals()
        assert len(stop_state.signal_history) == 1
    print("PASS  test_warmup_guard")


def test_max_rounds_cap():
    """Pipeline stops at max_rounds even if stopping conditions not met."""
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 2
        max_rounds = 3
        pools    = build_synthetic_pools(n_images=n_images, n_rounds=max_rounds)
        img_ids, imgs = make_stub_images(n_images)

        cfg = make_config(tmp, max_rounds=max_rounds, warmup_rounds=1)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        _, _, stop_state = orc.run("fake")
        assert stop_state.stopped
        assert stop_state.stop_reason == "max_rounds"
    print("PASS  test_max_rounds_cap")


def test_bank_resumable():
    """Reloading bank after run gives same committed rounds."""
    with tempfile.TemporaryDirectory() as tmp:
        n_images = 2
        pools    = build_synthetic_pools(n_images=n_images, n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1)
        orc = Orchestrator(
            config        = cfg,
            trainer       = StubTrainer(),
            infer_fn      = make_stub_infer_fn(pools),
            get_images_fn = lambda: (img_ids, imgs),
            load_model_fn = make_stub_load_model(),
        )
        orc.run("fake")

        # Reload from same path
        bank2 = PseudoLabelBank.load_or_create(tmp)
        assert bank2.committed_rounds() == [0, 1, 2]
        pool_r1 = bank2.get_candidate_pool(round_id=1)
        assert len(pool_r1) == n_images
    print("PASS  test_bank_resumable")


def test_config_tau_pre_mismatch_raises():
    """PipelineConfig must raise if tau_pre != min_iou_gate."""
    try:
        cfg = PipelineConfig(
            bank_path = "tmp",
            matching  = MatchingConfig(min_iou_gate=0.3),
            scoring   = ScoringConfig(tau_pre=0.5),  # mismatch
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "tau_pre" in str(e)
    print("PASS  test_config_tau_pre_mismatch_raises")


def test_jitter_all_pass_sets_jitter_executed():
    """
    When all three jitter passes return IoU >= min_iou_gate,
    the scorer must see jitter_executed=True and c_jitter_loc != 0.0.
    Verified via raw bank JSON.
    """
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        n_images = 1
        pools    = build_synthetic_pools(n_images=n_images, n_boxes=2, n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        # Jitter stub: always returns a valid IoU above gate=0.1
        def stub_jitter_fn(model, image, pl_t, transform, config):
            return 0.85  # always above min_iou_gate=0.1

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1,
                          tau_stab=0.0, tau_conf=0.0)
        orc = Orchestrator(
            config          = cfg,
            trainer         = StubTrainer(),
            infer_fn        = make_stub_infer_fn(pools),
            get_images_fn   = lambda: (img_ids, imgs),
            load_model_fn   = make_stub_load_model(),
            jitter_infer_fn = stub_jitter_fn,
        )
        orc.run("fake")

        # Check round 2 bank entries — round 1 has no prev so no matching
        round_path = Path(tmp) / "round_0002.json"
        assert round_path.exists()
        with open(round_path) as f:
            entries = json.load(f)

        # At least one matched box should have jitter_executed=True
        any_executed = any(
            any(v for v in entry["jitter_executed"].values())
            for entry in entries
        )
        assert any_executed, (
            "No jitter_executed=True in round 2 despite all jitter passes valid")

    print("PASS  test_jitter_all_pass_sets_jitter_executed")


def test_jitter_one_fail_omits_from_jitter_ious():
    """
    When one jitter pass returns 0.0 (below min_iou_gate),
    the box must be OMITTED from jitter_ious.
    Scorer must see jitter_executed=False and c_jitter_loc=0.0 (not 1.0).
    """
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        n_images = 1
        pools    = build_synthetic_pools(n_images=n_images, n_boxes=2, n_rounds=2)
        img_ids, imgs = make_stub_images(n_images)

        # Jitter stub: first call per box returns 0.0 (fails gate)
        call_counts = {}
        def stub_jitter_fn_fail_first(model, image, pl_t, transform, config):
            key = pl_t.box_id
            call_counts[key] = call_counts.get(key, 0) + 1
            # scale pass (first call) always fails
            if transform == "scale":
                return 0.0  # below min_iou_gate=0.1
            return 0.85

        cfg = make_config(tmp, max_rounds=2, warmup_rounds=1,
                          tau_stab=0.0, tau_conf=0.0)
        orc = Orchestrator(
            config          = cfg,
            trainer         = StubTrainer(),
            infer_fn        = make_stub_infer_fn(pools),
            get_images_fn   = lambda: (img_ids, imgs),
            load_model_fn   = make_stub_load_model(),
            jitter_infer_fn = stub_jitter_fn_fail_first,
        )
        orc.run("fake")

        round_path = Path(tmp) / "round_0002.json"
        with open(round_path) as f:
            entries = json.load(f)

        # No box should have jitter_executed=True (scale fails → all omitted)
        any_executed = any(
            any(v for v in entry["jitter_executed"].values())
            for entry in entries
        )
        assert not any_executed, (
            "jitter_executed=True despite scale pass failing — "
            "jitter triple should be omitted when any pass fails")

        # All matched boxes should have c_jitter_loc=0.0 exactly
        # (not None — None means unmatched/ambiguous, which is different)
        for entry in entries:
            for box_id, c_jitter in entry["c_jitter_loc"].items():
                score = entry["stability_scores"].get(box_id)
                if score is not None:
                    # matched box — jitter omitted → must be exactly 0.0
                    assert c_jitter == 0.0, (
                        f"box {box_id}: c_jitter_loc={c_jitter}, "
                        f"expected exactly 0.0 when jitter triple omitted"
                    )

    print("PASS  test_jitter_one_fail_omits_from_jitter_ious")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_two_round_loop_runs,
        test_round_metadata_written_for_all_rounds,
        test_bank_entries_fully_populated,
        test_admitted_is_subset_of_c_t,
        test_signal_history_populated,
        test_warmup_guard,
        test_max_rounds_cap,
        test_bank_resumable,
        test_config_tau_pre_mismatch_raises,
        test_jitter_all_pass_sets_jitter_executed,
        test_jitter_one_fail_omits_from_jitter_ious,
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
    print(f"Phase 6A smoke tests: {passed} passed, {failed} failed")
    print("=" * 50)