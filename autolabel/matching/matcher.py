"""
matching/matcher.py
===================
Hungarian matching for one image, one round-pair (t, t-1).

Architecture constraints (frozen):
  - Hungarian assignment, not greedy IoU
  - Cost: (1 - IoU(b_i, b_j)) + lambda_cls * JS(p_i, p_j)
  - Hard IoU gate: pair rejected if IoU < min_iou_gate
  - Ambiguity rejection: if cost gap between best and second-best match
    for a box is below epsilon_match, that box is marked ambiguous and
    excluded from stability scoring entirely
  - Single source of box identity for both scorer and stopping evaluator
  - CPU only
  - All IoU calls through utils/iou.py
  - All JS calls through utils/js_divergence.py

Ambiguous-box accounting (frozen Patch 4):
  Ambiguous current boxes are NOT in matched_pairs.
  Their nearest previous-round boxes are therefore absent from matched_pairs
  and will be counted as removed in RawChurn — conservative overestimate,
  biasing stopper toward not terminating early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from autolabel.bank.schemas import MatchResult, PseudoLabel
from autolabel.utils.iou import iou_pair
from autolabel.utils.js_divergence import js_divergence, normalize


@dataclass
class MatchingConfig:
    """
    Configuration for Hungarian matching.
    All fields must be identical across all rounds.

    lambda_cls:    weight for JS term in cost (default 1.0)
    min_iou_gate:  minimum IoU for a valid pair (default 0.3)
                   shared with jitter selection rule and tau_pre —
                   single key prevents silent divergence
    epsilon_match: cost-gap threshold for ambiguity rejection (default 0.05)
    """
    lambda_cls:    float = 1.0
    min_iou_gate:  float = 0.3
    epsilon_match: float = 0.05


class HungarianMatcher:
    """
    Matches detections from round t against round t-1 for a single image.
    Returns a MatchResult that is the single source of box identity for
    both the scorer and the stopping evaluator.
    """

    def __init__(self, config: MatchingConfig):
        self.config = config

    def match(self,
              current:  List[PseudoLabel],
              previous: List[PseudoLabel],
              image_id: str = "",
              round_t:  int = 0) -> MatchResult:
        """
        Match current-round detections against previous-round detections.

        All current box_ids appear in exactly one of:
            matched_pairs (as first element), unmatched_current, ambiguous.
        """
        # STEP 1: Edge cases
        if len(current) == 0:
            return MatchResult(
                image_id=image_id, round_t=round_t,
                matched_pairs=[], unmatched_current=[], ambiguous=[])

        if len(previous) == 0:
            return MatchResult(
                image_id=image_id, round_t=round_t,
                matched_pairs=[],
                unmatched_current=[pl.box_id for pl in current],
                ambiguous=[])

        N = len(current)
        M = len(previous)

        # STEP 2: Cost and IoU matrices [N x M]
        iou_mat  = np.zeros((N, M), dtype=np.float64)
        cost_mat = np.zeros((N, M), dtype=np.float64)

        for i, pl_i in enumerate(current):
            p_i = normalize(np.array(pl_i.class_scores, dtype=np.float64))
            for j, pl_j in enumerate(previous):
                iou_val       = iou_pair(pl_i.box, pl_j.box)
                p_j           = normalize(np.array(pl_j.class_scores,
                                                    dtype=np.float64))
                js_val        = js_divergence(p_i, p_j)
                iou_mat[i, j] = iou_val
                cost_mat[i, j]= (1.0 - iou_val) + self.config.lambda_cls * js_val

        # STEP 3: Hungarian assignment on GATED cost matrix
        # Apply IoU gate BEFORE Hungarian so the solver chooses among valid
        # candidates. Without this, Hungarian may choose an invalid pair
        # (IoU < gate) that has a low semantic cost, leaving a valid pair
        # unselected — then hard-gate filtering removes the invalid pair and
        # the current box becomes spuriously unmatched.
        LARGE_COST = 1e9
        cost_gated = cost_mat.copy()
        cost_gated[iou_mat < self.config.min_iou_gate] = LARGE_COST
        row_ind, col_ind = linear_sum_assignment(cost_gated)

        # STEP 4: Hard-gate filtering — reject any pair still at LARGE_COST
        # (current box had no valid previous candidate at all)
        valid_pairs: List[Tuple[int, int]] = [
            (r, c) for r, c in zip(row_ind, col_ind)
            if iou_mat[r, c] >= self.config.min_iou_gate
        ]

        # STEP 5: Ambiguity rejection — gap computed over VALID candidates only
        # A box is ambiguous only if two or more geometrically valid previous
        # boxes are equally plausible. An invalid candidate (IoU < gate) must
        # not contribute to the gap — that would mark a box ambiguous because
        # of a geometrically impossible match, which is not principled.
        ambiguous_rows: set = set()
        for r, _ in valid_pairs:
            valid_cols  = [j for j in range(M)
                           if iou_mat[r, j] >= self.config.min_iou_gate]
            if len(valid_cols) < 2:
                # Only one valid candidate — cannot be ambiguous
                continue
            valid_costs = sorted(cost_mat[r, j] for j in valid_cols)
            best        = valid_costs[0]
            second      = valid_costs[1]
            gap         = second - best
            if gap < self.config.epsilon_match:
                ambiguous_rows.add(r)

        final_pairs: List[Tuple[int, int]] = [
            (r, c) for r, c in valid_pairs
            if r not in ambiguous_rows
        ]

        # STEP 6: Classify
        matched_rows      = {r for r, _ in final_pairs}
        matched_pairs     = [(current[r].box_id, previous[c].box_id)
                             for r, c in final_pairs]
        unmatched_current = [current[r].box_id
                             for r in range(N)
                             if r not in matched_rows
                             and r not in ambiguous_rows]
        ambiguous         = [current[r].box_id for r in ambiguous_rows]

        return MatchResult(
            image_id=image_id, round_t=round_t,
            matched_pairs=matched_pairs,
            unmatched_current=unmatched_current,
            ambiguous=ambiguous,
        )