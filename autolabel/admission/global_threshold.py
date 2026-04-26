"""
admission/global_threshold.py
==============================
MVP AdmissionPolicy: global threshold on stability score and confidence.

Architecture constraints (frozen):
  - Admits a box if S(p,t) >= tau_stab AND q_t >= tau_conf
  - Both conditions must hold — AND, not OR
  - Global thresholds only — same tau_stab and tau_conf for all classes
  - No per-class logic (explicitly out of scope for MVP)
  - Boxes with score=None (unmatched/ambiguous) are never admitted

Known limitation (to be stated in paper):
  At 1% labeled data, tail-class AP will likely underperform relative to
  a class-aware admission version. This is accepted for MVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from autolabel.admission.interface import AdmissionPolicy
from autolabel.bank.schemas import CandidatePool, PseudoLabel


@dataclass
class AdmissionConfig:
    """
    Configuration for global threshold admission.

    tau_stab: minimum stability score S(p,t) to admit (default 0.6)
    tau_conf: minimum detection confidence q_t to admit (default 0.4)
              q_t is results.boxes.conf — admission gate ONLY
    """
    tau_stab: float = 0.6
    tau_conf: float = 0.4


class GlobalThresholdPolicy(AdmissionPolicy):
    """
    MVP admission policy using global score and confidence thresholds.

    Admits PseudoLabel pl if:
        scores[pl.box_id] is not None          (matched, not ambiguous)
        AND scores[pl.box_id] >= tau_stab      (stable enough)
        AND pl.confidence >= tau_conf          (confident enough)
    """

    def __init__(self, config: AdmissionConfig):
        self.config = config

    def admit(self,
              candidates: CandidatePool,
              scores:     Dict[str, Optional[float]],
              round_id:   int) -> List[PseudoLabel]:
        """
        Admit pseudo-labels passing both gates.

        Args:
            candidates: C_t — full candidate pool
            scores:     ScoringResult.scores — box_id → S or None
            round_id:   current round (unused in MVP, available for logging)

        Returns:
            List[PseudoLabel] admitted to A_t, ordered by image then detection.
        """
        admitted: List[PseudoLabel] = []

        for image_id in sorted(candidates):
            for pl in candidates[image_id]:
                s = scores.get(pl.box_id)
                if s is None:
                    # Unmatched or ambiguous — never admit
                    continue
                if s >= self.config.tau_stab and pl.confidence >= self.config.tau_conf:
                    admitted.append(pl)

        return admitted