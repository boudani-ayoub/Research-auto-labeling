"""
bank/schemas.py
===============
Single source of truth for every dataclass used in the pipeline.

Rules enforced here:
  - No computation anywhere in this file
  - Every dataclass the architecture spec names is defined exactly once here
  - Frozen types are frozen; mutable types are not frozen
  - CandidatePool and CandidateIndex are defined here, not in orchestrator

Patch 1 contracts (q_t separation):
  PseudoLabel.confidence   = results.boxes.conf  → admission gate ONLY
  PseudoLabel.class_scores = softmax(logits)     → C_cls_dist and JS cost ONLY
  These two fields are never assumed numerically equal anywhere in the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Core detection type ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PseudoLabel:
    """
    One post-NMS predicted box from canonical inference.

    confidence:
        q_t, taken directly from results.boxes.conf (Ultralytics post-NMS
        detection score). Role: admission confidence gate ONLY.
        Never assumed equal to max(class_scores).

    class_scores:
        Full K-dimensional softmax vector recovered via the Section 10
        extraction path (ClassScoreCapturingPredictor). Length == num_classes,
        sums to 1.0. Role: C_cls_dist and JS matching cost ONLY.
        Never used as admission gate directly.

    box_id convention:
        f"{image_id}_r{round_id}_{det_index}"
        Uniquely identifies a detection across all rounds and images.
    """
    image_id:     str
    box_id:       str                               # f"{image_id}_r{round_id}_{det_index}"
    round_id:     int
    box:          Tuple[float, float, float, float] # (x1, y1, x2, y2) absolute pixels
    pred_class:   int
    class_scores: Tuple[float, ...]                 # length K, sums to 1.0
    confidence:   float                             # q_t — from results.boxes.conf


# ── Pool and index types ──────────────────────────────────────────────────────

# Canonical pool type — used everywhere C_t or C_prev appears in the spec.
# image_id → list of post-NMS detections for that image.
CandidatePool = Dict[str, List[PseudoLabel]]


@dataclass
class CandidateIndex:
    """
    Fast by-id lookup over a CandidatePool.
    Built once per round in the orchestrator, never mutated after construction.
    """
    _index: Dict[str, PseudoLabel] = field(default_factory=dict)

    @staticmethod
    def build(pool: CandidatePool) -> "CandidateIndex":
        """Build index from a CandidatePool. O(n) over all detections."""
        idx = CandidateIndex()
        for pls in pool.values():
            for pl in pls:
                idx._index[pl.box_id] = pl
        return idx

    def get(self, box_id: str) -> Optional[PseudoLabel]:
        """Return PseudoLabel for box_id, or None if not present."""
        return self._index.get(box_id)

    def __contains__(self, box_id: str) -> bool:
        return box_id in self._index

    def __len__(self) -> int:
        return len(self._index)


# ── Matching output ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatchResult:
    """
    Hungarian matching output for one image, one round-pair (t, t-1).

    Single source of box identity for both scorer and stopping evaluator —
    no module may perform its own independent box-identity logic.

    Ambiguous boxes (cost gap < epsilon_match) are NOT in matched_pairs.
    Their nearest previous-round boxes are therefore absent from matched_pairs
    and are counted as removed in RawChurn. This is conservative accounting:
    it slightly overestimates churn, biasing the stopper toward not terminating
    early. See frozen design decision in architecture spec.

    Fields:
        image_id:          image this result covers
        round_t:           current round index
        matched_pairs:     list of (box_id_t, box_id_prev) valid pairs
        unmatched_current: box_ids in C_t with no valid previous match
                           (excludes ambiguous — those are in .ambiguous)
        ambiguous:         box_ids in C_t excluded due to cost-gap < epsilon_match
    """
    image_id:          str
    round_t:           int
    matched_pairs:     List[Tuple[str, str]]  # (box_id_t, box_id_prev)
    unmatched_current: List[str]
    ambiguous:         List[str]


# ── Scoring types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentRecord:
    """
    Per-box component evidence for one detection at round t.
    Stored in bank for ablations and debugging.

    c_cls_dist:
        1 - JS(class_scores_t, class_scores_prev) / ln(2).
        None if box is unmatched or ambiguous.

    c_round_loc:
        IoU(box_t, box_prev) on canonical images.
        None if box is unmatched or ambiguous.

    c_jitter_loc:
        1 - population_std([IoU_1, IoU_2, IoU_3]) across three jitter passes.
        0.0 if box did not pass the prefilter (c_round_loc < tau_pre).
        None if box is unmatched or ambiguous.
        NOTE: 0.0 and None mean different things — do not conflate.

    jitter_executed:
        True  if all three jitter passes ran.
        False if prefilter not passed, or box is unmatched/ambiguous.
    """
    c_cls_dist:      Optional[float]  # None if unmatched or ambiguous
    c_round_loc:     Optional[float]  # None if unmatched or ambiguous
    c_jitter_loc:    Optional[float]  # 0.0 if prefilter failed; None if unmatched
    jitter_executed: bool             # False if prefilter not passed or unmatched


@dataclass(frozen=True)
class ScoringResult:
    """
    Single return type from scorer.score().
    Unpacked exactly once in the orchestrator — no separate getter methods.

    scores:
        box_id → S(p,t) in [0,1], or None if box is unmatched/ambiguous.

    components:
        box_id → ComponentRecord with per-component evidence.
        Present for every box in C_t, including unmatched and ambiguous.
    """
    scores:     Dict[str, Optional[float]]  # box_id → S(p,t) or None
    components: Dict[str, ComponentRecord]  # box_id → component evidence


# ── Bank storage types ────────────────────────────────────────────────────────

@dataclass
class BankEntry:
    """
    One image's complete record for one round.
    Append-only — never overwritten after being written to disk.

    All dict fields are keyed by box_id for O(1) lookup.
    Every box in pseudo_labels must appear in every dict field.
    """
    image_id:            str
    round_id:            int
    pseudo_labels:       List[PseudoLabel]
    stability_scores:    Dict[str, Optional[float]]  # box_id → S or None
    admitted:            Dict[str, bool]              # box_id → admitted?
    matched_prev_box_id: Dict[str, Optional[str]]    # box_id → prev box_id or None
    is_ambiguous:        Dict[str, bool]              # box_id → ambiguous?
    c_cls_dist:          Dict[str, Optional[float]]
    c_round_loc:         Dict[str, Optional[float]]
    c_jitter_loc:        Dict[str, Optional[float]]
    jitter_executed:     Dict[str, bool]


@dataclass
class RoundMetadata:
    """
    Summary record written at the end of each round.
    Includes stopping signal values for that round.
    """
    round_id:           int
    n_candidates:       int    # |C_t|
    n_admitted:         int    # |A_t|
    raw_churn:          float
    stable_yield:       float
    class_drift:        float
    stop_condition_met: bool
    model_checkpoint:   str    # path to YOLOv8 .pt file used this round


# ── Stopping types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StoppingSnapshot:
    """
    One round's stopping signals.

    signal_history in StoppingState is always List[StoppingSnapshot] —
    never raw tuples. Fields are always accessed by name (.raw_churn etc),
    never by index. This was a frozen design decision added to kill an earlier
    spec bug where raw tuples caused silent field-order errors.
    """
    round_id:     int
    raw_churn:    float
    stable_yield: float
    class_drift:  float


@dataclass
class StoppingState:
    """
    Mutable state carried across rounds by the stopping evaluator.

    consecutive_satisfied:
        Number of consecutive rounds where all three stopping conditions
        held simultaneously. Reset to 0 on any round where conditions fail.

    signal_history:
        Ordered list of StoppingSnapshot, one per round processed.
        Warmup rounds have snapshots appended but do not affect the counter.

    stopped:
        Set to True when stopping fires (convergence or max_rounds).

    stop_reason:
        'convergence' — K_consecutive rounds of satisfied conditions
        'max_rounds'  — hard cap hit
        None          — not stopped yet
    """
    consecutive_satisfied: int                    = 0
    signal_history:        List[StoppingSnapshot] = field(default_factory=list)
    stopped:               bool                   = False
    stop_reason:           Optional[str]          = None  # 'convergence'|'max_rounds'|None