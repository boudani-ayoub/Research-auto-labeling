"""
stopping/stopping.py
====================
Stopping signals and stopping rule for the iterative auto-labeling pipeline.

Architecture constraints (frozen):
  - Identity source: MatchResult from matcher — no independent identity logic
  - Signals computed over C_t, NEVER over A_t (decoupling is mandatory)
  - signal_history is always List[StoppingSnapshot] — never raw tuples
  - Fields accessed by name (.raw_churn, etc.) — never by index
  - Warmup guard and max_rounds cap are enforced by the ORCHESTRATOR, not here
  - StabilityScorer provides scores; this module only evaluates stopping

RawChurn ambiguous-box accounting (frozen Patch 4):
  Ambiguous current boxes are NOT in matched_pairs.
  Their nearest previous-round boxes are therefore absent from matched_pairs
  and are counted as removed. This overestimates churn — the correct bias
  for a system where early stopping is the worse failure mode.

ClassDrift: JS(marginal_t, marginal_prev) already normalized by ln(2)
  via js_divergence_marginals() in utils/js_divergence.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from autolabel.bank.schemas import (
    CandidatePool,
    MatchResult,
    PseudoLabel,
    StoppingSnapshot,
    StoppingState,
)
from autolabel.utils.iou import iou_pair
from autolabel.utils.js_divergence import js_divergence_marginals


@dataclass
class StoppingConfig:
    """
    Thresholds and parameters for the stopping rule.
    All fields must be identical across all rounds.

    epsilon_churn:   max |RawChurn_t - RawChurn_{t-1}| to satisfy churn cond
    epsilon_yield:   max |StableYield_t - StableYield_{t-1}| for yield cond
    epsilon_drift:   max |ClassDrift_t - ClassDrift_{t-1}| for drift cond
    tau_yield:       floor threshold — yield cond satisfied if below this
    tau_drift:       ceiling threshold — drift cond satisfied if above this
    tau_geom_change: IoU threshold for counting a matched pair as geom-changed
    K_consecutive:   rounds of consecutive satisfaction required to stop
    """
    epsilon_churn:   float = 0.02
    epsilon_yield:   float = 0.03
    epsilon_drift:   float = 0.01
    tau_yield:       float = 0.05
    tau_drift:       float = 0.10
    tau_geom_change: float = 0.50
    K_consecutive:   int   = 2


class StoppingEvaluator:
    """
    Computes stopping signals and evaluates the stopping rule.

    Warmup and max_rounds are the orchestrator's responsibility.
    This class only computes signals and updates StoppingState.

    Usage pattern in orchestrator:
        if round_t > warmup_rounds:
            stop_state = stopper.evaluate(C_t, C_prev, A_t,
                                          match_results, stop_state, round_t)
        else:
            snapshot = stopper.compute_signals(C_t, C_prev, A_t,
                                               match_results, round_t)
            stop_state.signal_history.append(snapshot)
    """

    def __init__(self, config: StoppingConfig):
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self,
                 C_t:           CandidatePool,
                 C_prev:        CandidatePool,
                 A_t:           List[PseudoLabel],
                 match_results: Dict[str, MatchResult],
                 stop_state:    StoppingState,
                 round_t:       int) -> StoppingState:
        """
        Compute signals, append snapshot, evaluate conditions, update state.

        Called by orchestrator for non-warmup rounds only.
        Returns updated StoppingState (mutated in place and returned).
        """
        # STEP 4: Append snapshot (validation happens inside compute_signals)
        snapshot = self.compute_signals(C_t, C_prev, A_t, match_results, round_t)
        stop_state.signal_history.append(snapshot)

        # STEP 5: Conditions — requires >= 2 snapshots for deltas
        if len(stop_state.signal_history) < 2:
            return stop_state

        prev = stop_state.signal_history[-2]

        cond_churn = (
            abs(snapshot.raw_churn - prev.raw_churn) < self.config.epsilon_churn
        )
        cond_yield = (
            abs(snapshot.stable_yield - prev.stable_yield) < self.config.epsilon_yield
            or snapshot.stable_yield < self.config.tau_yield
        )
        cond_drift = (
            abs(snapshot.class_drift - prev.class_drift) < self.config.epsilon_drift
            or snapshot.class_drift > self.config.tau_drift
        )

        # STEP 6: Counter
        if cond_churn and cond_yield and cond_drift:
            stop_state.consecutive_satisfied += 1
        else:
            stop_state.consecutive_satisfied = 0

        # STEP 7: Fire
        if stop_state.consecutive_satisfied >= self.config.K_consecutive:
            stop_state.stopped     = True
            stop_state.stop_reason = "convergence"

        return stop_state

    def compute_signals(self,
                        C_t:           CandidatePool,
                        C_prev:        CandidatePool,
                        A_t:           List[PseudoLabel],
                        match_results: Dict[str, MatchResult],
                        round_t:       int) -> StoppingSnapshot:
        """
        Compute all three stopping signals for round_t.
        Returns a StoppingSnapshot — does NOT mutate stop_state.
        Called directly for warmup rounds.
        """
        # Validate consistency — silent corruption not acceptable
        self._validate_match_results(C_t, C_prev, match_results)
        self._validate_a_t(A_t, C_t)

        n_t    = sum(len(v) for v in C_t.values())
        n_prev = sum(len(v) for v in C_prev.values())

        raw_churn    = self._compute_raw_churn(C_t, C_prev, match_results, n_t, n_prev)
        stable_yield = self._compute_stable_yield(A_t, n_t)
        class_drift  = self._compute_class_drift(C_t, C_prev, n_t, n_prev)

        return StoppingSnapshot(
            round_id     = round_t,
            raw_churn    = raw_churn,
            stable_yield = stable_yield,
            class_drift  = class_drift,
        )

    # ── Signal computations ───────────────────────────────────────────────────

    def _compute_raw_churn(self,
                           C_t:           CandidatePool,
                           C_prev:        CandidatePool,
                           match_results: Dict[str, MatchResult],
                           n_t:           int,
                           n_prev:        int) -> float:
        """
        RawChurn_t = (N_new + N_removed + N_class_change + N_geom_change)
                     / |C_t ∪ C_{t-1}|

        All identity comes from match_results — no independent box-identity logic.

        N_removed counts prev boxes absent from matched_pairs, including those
        nearest to ambiguous current boxes (Patch 4 conservative accounting).

        NOTE: RawChurn is an event-rate signal, not a probability. A single
        matched pair can count in both N_class_change and N_geom_change, so
        RawChurn can exceed 1.0. This is intentional — it is not clamped.
        The stopping conditions use deltas, so the absolute scale only
        matters for interpretation, not for correctness of the logic.
        """
        n_matched = sum(len(mr.matched_pairs) for mr in match_results.values())
        union_size = n_t + n_prev - n_matched

        if union_size == 0:
            return 0.0

        # N_new: unmatched current boxes (new detections)
        n_new = sum(len(mr.unmatched_current) for mr in match_results.values())

        # N_removed: prev boxes not in any matched_pair
        # Includes prev boxes nearest to ambiguous current boxes (Patch 4)
        n_removed = 0
        for image_id, mr in match_results.items():
            matched_prev = {bp for _, bp in mr.matched_pairs}
            n_removed += sum(
                1 for pl in C_prev.get(image_id, [])
                if pl.box_id not in matched_prev
            )

        # N_class_change: matched pairs where pred_class changed
        n_class_change = 0
        for image_id, mr in match_results.items():
            prev_pls = {pl.box_id: pl for pl in C_prev.get(image_id, [])}
            curr_pls = {pl.box_id: pl for pl in C_t.get(image_id, [])}
            for box_id_t, box_id_prev in mr.matched_pairs:
                pl_t    = curr_pls.get(box_id_t)
                pl_prev = prev_pls.get(box_id_prev)
                if pl_t and pl_prev and pl_t.pred_class != pl_prev.pred_class:
                    n_class_change += 1

        # N_geom_change: matched pairs where IoU < tau_geom_change
        n_geom_change = 0
        for image_id, mr in match_results.items():
            prev_pls = {pl.box_id: pl for pl in C_prev.get(image_id, [])}
            curr_pls = {pl.box_id: pl for pl in C_t.get(image_id, [])}
            for box_id_t, box_id_prev in mr.matched_pairs:
                pl_t    = curr_pls.get(box_id_t)
                pl_prev = prev_pls.get(box_id_prev)
                if pl_t and pl_prev:
                    iou = iou_pair(pl_t.box, pl_prev.box)
                    if iou < self.config.tau_geom_change:
                        n_geom_change += 1

        numerator = n_new + n_removed + n_class_change + n_geom_change
        return float(numerator / union_size)

    def _compute_stable_yield(self,
                               A_t: List[PseudoLabel],
                               n_t: int) -> float:
        """
        StableYield_t = |A_t| / |C_t|

        Computed over C_t size, not A_t size — uses n_t passed from caller.
        Returns 0.0 if C_t is empty.
        """
        if n_t == 0:
            return 0.0
        return float(len(A_t) / n_t)

    def _compute_class_drift(self,
                              C_t:    CandidatePool,
                              C_prev: CandidatePool,
                              n_t:    int,
                              n_prev: int) -> float:
        """
        ClassDrift_t = JS(marginal_t, marginal_prev) normalized by ln(2).

        If C_prev is empty, uses uniform distribution as marginal_prev.
        Determines num_classes from whichever pool is non-empty.
        """
        # Determine num_classes from class_scores length
        num_classes = self._get_num_classes(C_t, C_prev)
        if num_classes == 0:
            return 0.0

        counts_t    = self._class_marginal(C_t,    num_classes)
        counts_prev = self._class_marginal(C_prev, num_classes)

        if n_prev == 0:
            # No previous pool — use uniform distribution
            counts_prev = np.ones(num_classes, dtype=np.float64)

        return float(js_divergence_marginals(counts_t, counts_prev))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate_match_results(self,
                                 C_t:           CandidatePool,
                                 C_prev:        CandidatePool,
                                 match_results: Dict[str, MatchResult]) -> None:
        """
        Validate that match_results is consistent with C_t and C_prev.
        Raises ValueError on any inconsistency.

        Checks:
          - Every image in C_t union C_prev has a MatchResult
          - Every current box_id in matched_pairs, unmatched_current, ambiguous
            exists in C_t for that image
          - Every previous box_id in matched_pairs exists in C_prev
        """
        all_images = set(C_t.keys()) | set(C_prev.keys())
        for image_id in all_images:
            if image_id not in match_results:
                raise ValueError(
                    f"Missing MatchResult for image {image_id!r}. "
                    f"match_results must cover every image in C_t union C_prev."
                )
        for image_id, mr in match_results.items():
            curr_ids = {pl.box_id for pl in C_t.get(image_id, [])}
            prev_ids = {pl.box_id for pl in C_prev.get(image_id, [])}
            for box_id_t, box_id_prev in mr.matched_pairs:
                if box_id_t not in curr_ids:
                    raise ValueError(
                        f"matched_pairs current box {box_id_t!r} "
                        f"(image {image_id!r}) not found in C_t."
                    )
                if box_id_prev not in prev_ids:
                    raise ValueError(
                        f"matched_pairs previous box {box_id_prev!r} "
                        f"(image {image_id!r}) not found in C_prev."
                    )
            for box_id in mr.unmatched_current:
                if box_id not in curr_ids:
                    raise ValueError(
                        f"unmatched_current box {box_id!r} "
                        f"(image {image_id!r}) not found in C_t."
                    )
            for box_id in mr.ambiguous:
                if box_id not in curr_ids:
                    raise ValueError(
                        f"ambiguous box {box_id!r} "
                        f"(image {image_id!r}) not found in C_t."
                    )

    def _validate_a_t(self,
                      A_t: List[PseudoLabel],
                      C_t: CandidatePool) -> None:
        """
        Validate A_t against C_t.
        Raises ValueError if:
          - any A_t box_id is not in C_t (ghost label)
          - any box_id appears more than once in A_t (duplicate)
        """
        all_ct_ids = {pl.box_id for pls in C_t.values() for pl in pls}
        seen: set = set()
        for pl in A_t:
            if pl.box_id not in all_ct_ids:
                raise ValueError(
                    f"A_t box {pl.box_id!r} not found in C_t. "
                    f"A_t must be a subset of C_t."
                )
            if pl.box_id in seen:
                raise ValueError(
                    f"Duplicate box_id {pl.box_id!r} in A_t."
                )
            seen.add(pl.box_id)

    def _get_num_classes(self,
                          C_t:    CandidatePool,
                          C_prev: CandidatePool) -> int:
        """Infer num_classes from the first available PseudoLabel."""
        for pool in (C_t, C_prev):
            for pls in pool.values():
                if pls:
                    return len(pls[0].class_scores)
        return 0

    def _class_marginal(self,
                         pool:        CandidatePool,
                         num_classes: int) -> np.ndarray:
        """
        Count detections per class across all images in pool.
        Returns integer count vector of length num_classes.
        """
        counts = np.zeros(num_classes, dtype=np.float64)
        for pls in pool.values():
            for pl in pls:
                counts[pl.pred_class] += 1
        return counts