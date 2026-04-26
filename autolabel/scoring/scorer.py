"""
scoring/scorer.py
=================
Stability scoring for all boxes in C_t.

Architecture constraints (frozen):
  - Receives C_t, C_prev, match_results, jitter_ious from orchestrator
  - Returns a single ScoringResult — unpacked once in orchestrator, no getters
  - CPU only — no GPU tensors enter this module
  - Scorer only receives PRECOMPUTED jitter IoU triples — it does not run
    jitter inference itself (orchestrator owns that end-to-end)
  - All IoU calls through utils/iou.py
  - All JS calls through utils/js_divergence.py

Stability score formula (frozen):
  S(p,t) = clamp(alpha * C_cls_dist + beta * C_round_loc + gamma * C_jitter_loc, 0, 1)

Component definitions (frozen):
  C_cls_dist    = 1 - JS(normalize(class_scores_t), normalize(class_scores_prev))
                  JS already normalized by ln(2) in js_divergence.py → result in [0,1]
  C_round_loc   = IoU(box_t, box_prev)
  C_jitter_loc  = 1 - population_std([iou1, iou2, iou3])  (ddof=0)
                  If box_id_t absent from jitter_ious → C_jitter_loc = 0.0
                  No gamma redistribution when jitter unavailable (frozen)

Jitter convention (frozen — must be respected by orchestrator):
  jitter_ious is populated ONLY when all three jitter passes produced a
  valid same-class detection above min_iou_gate. If any pass fails (no
  same-class detection, or IoU below gate), the orchestrator OMITS that
  box from jitter_ious entirely — it does NOT pass (0, 0, 0).
  Rationale: std([0,0,0])=0 → C_jitter_loc=1.0, making a completely
  failed box look maximally stable. Omitting the box forces C_jitter_loc=0.0
  (the conservative, correct signal). This convention is enforced by the
  orchestrator's jitter_and_select logic, not here.

Null convention (frozen):
  Unmatched or ambiguous boxes → scores[box_id] = None
                                  components[box_id] = ComponentRecord(None, None, None, False)
  C_jitter_loc = 0.0 (prefilter failed) is DISTINCT from None (unmatched)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from autolabel.bank.schemas import (
    CandidateIndex,
    CandidatePool,
    ComponentRecord,
    MatchResult,
    ScoringResult,
)
from autolabel.utils.iou import iou_pair
from autolabel.utils.js_divergence import js_divergence, normalize


@dataclass
class ScoringConfig:
    """
    Weights and defaults for stability scoring.
    All fields must be identical across all rounds.

    alpha + beta + gamma should sum to 1.0 (not enforced, but expected).
    tau_pre is the prefilter threshold for jitter — must equal
    matching.min_iou_gate (asserted at config load time in orchestrator).
    """
    alpha:   float = 0.33
    beta:    float = 0.33
    gamma:   float = 0.34
    tau_pre: float = 0.3   # must equal matching.min_iou_gate


class StabilityScorer:
    """
    Computes stability scores for all boxes in C_t.

    Input:
        C_t:           CandidatePool — current round detections
        C_prev:        CandidatePool — previous round detections
        match_results: Dict[image_id, MatchResult] — from HungarianMatcher
        jitter_ious:   Dict[box_id_t, Tuple[float, float, float]]
                       Precomputed by orchestrator. Only present for boxes
                       that passed the c_round_loc >= tau_pre prefilter.

    Output:
        ScoringResult(scores, components)
        scores:     box_id → S(p,t) in [0,1], or None if unmatched/ambiguous
        components: box_id → ComponentRecord for every box in C_t
    """

    def __init__(self, config: ScoringConfig):
        self.config = config

    def score(self,
              C_t:           CandidatePool,
              C_prev:        CandidatePool,
              match_results: Dict[str, MatchResult],
              jitter_ious:   Dict[str, Tuple[float, float, float]]) -> ScoringResult:
        """
        Compute stability scores for all boxes in C_t.

        Returns ScoringResult — unpack once in orchestrator:
            scoring_result = scorer.score(...)
            scores_t     = scoring_result.scores
            components_t = scoring_result.components
        """
        # STEP 1: Build indices for O(1) lookup
        idx_t    = CandidateIndex.build(C_t)
        idx_prev = CandidateIndex.build(C_prev)

        scores:     Dict[str, Optional[float]]  = {}
        components: Dict[str, ComponentRecord]  = {}

        # STEP 2: Score matched pairs
        for mr in match_results.values():
            for box_id_t, box_id_prev in mr.matched_pairs:
                pl_t    = idx_t.get(box_id_t)
                pl_prev = idx_prev.get(box_id_prev)

                if pl_t is None or pl_prev is None:
                    # match_results must be consistent with C_t/C_prev.
                    # Silent None would hide upstream bugs — raise instead.
                    missing_t    = box_id_t    if pl_t    is None else None
                    missing_prev = box_id_prev if pl_prev is None else None
                    raise ValueError(
                        f"Inconsistent match_results: matched pair "
                        f"({box_id_t!r}, {box_id_prev!r}) references box_ids "
                        f"not found in C_t/C_prev. "
                        f"Missing in C_t: {missing_t!r}. "
                        f"Missing in C_prev: {missing_prev!r}. "
                        f"This is a pipeline bug — matcher output must be "
                        f"consistent with the pools passed to scorer."
                    )

                # C_cls_dist: 1 - JS(p_t, p_prev), JS already normalized by ln(2)
                p_t    = normalize(np.array(pl_t.class_scores,    dtype=np.float64))
                p_prev = normalize(np.array(pl_prev.class_scores, dtype=np.float64))
                c_cls_dist = 1.0 - js_divergence(p_t, p_prev)

                # C_round_loc: IoU between matched boxes
                c_round_loc = iou_pair(pl_t.box, pl_prev.box)

                # C_jitter_loc: 1 - population_std of jitter IoU triple
                if box_id_t in jitter_ious:
                    iou1, iou2, iou3 = jitter_ious[box_id_t]
                    # population std (ddof=0) — frozen spec
                    c_jitter_loc    = float(
                        1.0 - np.std([iou1, iou2, iou3], ddof=0))
                    jitter_executed = True
                else:
                    # Prefilter not passed (c_round_loc < tau_pre)
                    # Conservative: 0.0, no gamma redistribution
                    c_jitter_loc    = 0.0
                    jitter_executed = False

                # Stability score — clamped to [0, 1]
                s = (self.config.alpha * c_cls_dist
                     + self.config.beta  * c_round_loc
                     + self.config.gamma * c_jitter_loc)
                s = float(np.clip(s, 0.0, 1.0))

                scores[box_id_t]     = s
                components[box_id_t] = ComponentRecord(
                    c_cls_dist   = float(c_cls_dist),
                    c_round_loc  = float(c_round_loc),
                    c_jitter_loc = float(c_jitter_loc),
                    jitter_executed = jitter_executed,
                )

        # STEP 3: Null-fill unmatched and ambiguous boxes
        for mr in match_results.values():
            for box_id in mr.unmatched_current + mr.ambiguous:
                scores[box_id]     = None
                components[box_id] = ComponentRecord(
                    c_cls_dist=None, c_round_loc=None,
                    c_jitter_loc=None, jitter_executed=False)

        # STEP 4: Completeness check — every C_t box must be in output
        # If a C_t image is missing from match_results entirely, its boxes
        # will be absent from scores/components. That is a pipeline bug —
        # the orchestrator must pass match_results covering every image in C_t.
        for image_id, pls in C_t.items():
            for pl in pls:
                if pl.box_id not in scores:
                    raise ValueError(
                        f"C_t box {pl.box_id!r} (image {image_id!r}) is missing "
                        f"from scores output. Its image was not covered by "
                        f"match_results. The orchestrator must include every "
                        f"C_t image in match_results before calling scorer.score()."
                    )

        # STEP 5: Return single ScoringResult
        return ScoringResult(scores=scores, components=components)