"""
admission/interface.py
======================
AdmissionPolicy abstract base class.

Architecture constraints (frozen):
  - Only interface boundary between scoring and admission
  - admit() receives candidates (C_t), scores (from ScoringResult.scores),
    and round_id
  - Returns List[PseudoLabel] — the admitted subset A_t
  - MVP uses GlobalThresholdPolicy (global_threshold.py)
  - Class-aware admission is explicitly out of scope for MVP
    (known limitation: tail-class AP will underperform at 1% labeled data;
     must be stated in paper limitations section)
  - Strategy interface isolates admission so v2 can add class-aware logic
    without touching matching, scoring, or stopping
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from autolabel.bank.schemas import CandidatePool, PseudoLabel


class AdmissionPolicy(ABC):
    """
    Abstract base class for pseudo-label admission policies.

    Subclasses implement admit() to decide which candidates from C_t
    are admitted to A_t based on stability scores and any policy-specific
    logic.

    The interface is intentionally minimal — scoring and matching are
    complete before admit() is called, so the policy only needs to filter.
    """

    @abstractmethod
    def admit(self,
              candidates: CandidatePool,
              scores:     Dict[str, Optional[float]],
              round_id:   int) -> List[PseudoLabel]:
        """
        Select pseudo-labels to admit from C_t.

        Args:
            candidates: C_t — full candidate pool for this round
            scores:     ScoringResult.scores — box_id → S(p,t) or None
                        None means unmatched or ambiguous (never admitted)
            round_id:   current round (available for logging or per-round logic)

        Returns:
            List[PseudoLabel] — the admitted set A_t
            Must be a subset of boxes in candidates.
            Empty list is valid (no labels admitted this round).
        """
        ...