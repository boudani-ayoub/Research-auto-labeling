"""
autolabel/orchestrator.py
=========================
Top-level pipeline controller.

Architecture constraints (frozen):
  - Owns round sequencing: inference → retrieve → match → jitter → score
    → admit → bank → stop → metadata → train
  - Does NOT implement any of these operations
  - Enforces warmup guard and max_rounds cap unconditionally
  - Owns jitter inference end-to-end (Step 4)
  - ScoringResult is unpacked exactly once (Step 5)
  - Stopping computed over C_t, never A_t
  - Trainer is injectable — orchestrator does not import Ultralytics directly

Jitter selection rule (frozen, owned here):
  For each matched pair passing c_round_loc >= tau_pre:
    1. Apply transform to image; remap box via same geometry
    2. Run canonical_infer on jittered image (single image, batch_size=1)
    3. Filter detections to same pred_class as pl_t
    4. Select detection with highest IoU vs remapped base box
    5. If none qualifies (no same-class det or IoU < min_iou_gate): return 0.0
  Box omitted from jitter_ious if prefilter fails — never passed as (0,0,0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from autolabel.admission.global_threshold import AdmissionConfig, GlobalThresholdPolicy
from autolabel.admission.interface import AdmissionPolicy
from autolabel.bank.bank import PseudoLabelBank
from autolabel.bank.schemas import (
    BankEntry,
    CandidateIndex,
    CandidatePool,
    MatchResult,
    RoundMetadata,
    StoppingState,
)
from autolabel.matching.matcher import HungarianMatcher, MatchingConfig
from autolabel.scoring.scorer import ScoringConfig, StabilityScorer
from autolabel.stopping.stopping import StoppingConfig, StoppingEvaluator
from autolabel.utils.iou import iou_pair

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Top-level pipeline configuration. All sub-configs are nested here.
    """
    # Pipeline control
    max_rounds:    int   = 5
    warmup_rounds: int   = 2
    bank_path:     str   = "outputs/bank"

    # Sub-module configs
    matching:  MatchingConfig  = field(default_factory=MatchingConfig)
    scoring:   ScoringConfig   = field(default_factory=ScoringConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    stopping:  StoppingConfig  = field(default_factory=StoppingConfig)

    def __post_init__(self):
        # tau_pre must equal min_iou_gate — single key prevents silent divergence
        if self.scoring.tau_pre != self.matching.min_iou_gate:
            raise ValueError(
                f"Config violation: scoring.tau_pre ({self.scoring.tau_pre}) "
                f"must equal matching.min_iou_gate ({self.matching.min_iou_gate}). "
                f"These share one config key by design."
            )


# ── Trainer interface (injectable) ───────────────────────────────────────────

class TrainerInterface:
    """
    Abstract interface for the trainer. Orchestrator depends only on this.
    YOLOTrainer (Phase 6B) implements it; smoke tests use a stub.
    """
    def train(self,
              labeled_data:  object,
              pseudo_labels: list,
              round_id:      int) -> str:
        """Returns checkpoint path."""
        raise NotImplementedError


# ── Inference interfaces (injectable) ────────────────────────────────────────

InferFn  = Callable[[object, List[str], List[np.ndarray], int], CandidatePool]
ImagesFn = Callable[[], Tuple[List[str], List[np.ndarray]]]


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Runs the iterative auto-labeling pipeline.

    The orchestrator does not implement inference, matching, scoring,
    admission, stopping, or training. It sequences them.

    Args:
        config:         PipelineConfig
        trainer:        TrainerInterface implementation
        infer_fn:       callable(model, image_ids, images, round_id) → CandidatePool
        get_images_fn:  callable() → (image_ids, images) for unlabeled set
        load_model_fn:  callable(checkpoint_path) → model object
        jitter_infer_fn: callable(model, image, box, transform, config) → float
                         Returns IoU of best same-class detection on jittered image,
                         or 0.0 if none qualifies. Owned by orchestrator.
    """

    def __init__(self,
                 config:          PipelineConfig,
                 trainer:         TrainerInterface,
                 infer_fn:        InferFn,
                 get_images_fn:   ImagesFn,
                 load_model_fn:   Callable,
                 jitter_infer_fn: Optional[Callable] = None):
        self.config          = config
        self.trainer         = trainer
        self.infer_fn        = infer_fn
        self.get_images_fn   = get_images_fn
        self.load_model_fn   = load_model_fn
        self.jitter_infer_fn = jitter_infer_fn

        self.matcher  = HungarianMatcher(config.matching)
        self.scorer   = StabilityScorer(config.scoring)
        self.policy   = GlobalThresholdPolicy(config.admission)
        self.stopper  = StoppingEvaluator(config.stopping)
        self.bank     = PseudoLabelBank.load_or_create(config.bank_path)

    def run(self, labeled_data: object) -> Tuple[str, PseudoLabelBank, StoppingState]:
        """
        Run the full iterative pipeline.

        Returns:
            (final_checkpoint_path, bank, stop_state)
        """
        stop_state = StoppingState()

        # ── Round 0: supervised burn-in ───────────────────────────────────────
        logger.info("Round 0: supervised burn-in")
        checkpoint_t = self.trainer.train(
            labeled_data=labeled_data, pseudo_labels=[], round_id=0)
        self.bank.write_metadata(RoundMetadata(
            round_id=0, n_candidates=0, n_admitted=0,
            raw_churn=0.0, stable_yield=0.0, class_drift=0.0,
            stop_condition_met=False, model_checkpoint=checkpoint_t,
        ))

        # ── Main round loop ───────────────────────────────────────────────────
        for round_t in range(1, self.config.max_rounds + 1):
            logger.info(f"Round {round_t} / {self.config.max_rounds}")
            checkpoint_t, stop_state = self._run_round(
                round_t, checkpoint_t, labeled_data, stop_state)

            if stop_state.stopped:
                logger.info(
                    f"Stopping at round {round_t}: {stop_state.stop_reason}")
                break

        return checkpoint_t, self.bank, stop_state

    def _run_round(self,
                   round_t:      int,
                   checkpoint_t: str,
                   labeled_data: object,
                   stop_state:   StoppingState
                   ) -> Tuple[str, StoppingState]:
        """Execute one complete round. Returns (new_checkpoint, updated_stop_state)."""

        # STEP 1: Canonical inference
        model_t = self.load_model_fn(checkpoint_t)
        image_ids, images = self.get_images_fn()
        C_t   = self.infer_fn(model_t, image_ids, images, round_t)
        idx_t = CandidateIndex.build(C_t)

        # STEP 2: Retrieve prior-round candidates
        C_prev   = self.bank.get_candidate_pool(round_id=round_t - 1)
        idx_prev = CandidateIndex.build(C_prev)

        # STEP 3: Hungarian matching per image
        all_images   = set(C_t.keys()) | set(C_prev.keys())
        match_results: Dict[str, MatchResult] = {
            image_id: self.matcher.match(
                current  = C_t.get(image_id, []),
                previous = C_prev.get(image_id, []),
                image_id = image_id,
                round_t  = round_t,
            )
            for image_id in all_images
        }

        # STEP 4: Jitter inference (orchestrator owns entirely)
        jitter_ious: Dict[str, Tuple[float, float, float]] = {}
        if self.jitter_infer_fn is not None:
            jitter_ious = self._compute_jitter_ious(
                model_t, image_ids, images, match_results, idx_t, idx_prev)

        # STEP 5: Stability scoring — ScoringResult unpacked exactly once
        scoring_result = self.scorer.score(
            C_t=C_t, C_prev=C_prev,
            match_results=match_results,
            jitter_ious=jitter_ious,
        )
        scores_t     = scoring_result.scores
        components_t = scoring_result.components

        # STEP 6: Admission
        A_t = self.policy.admit(
            candidates=C_t, scores=scores_t, round_id=round_t)

        # STEP 7: Write to bank
        self._write_bank_entries(
            round_t, C_t, match_results, scores_t, components_t, A_t)

        # STEP 8: Stopping evaluation
        in_warmup = (round_t <= self.config.warmup_rounds)
        if not in_warmup:
            stop_state = self.stopper.evaluate(
                C_t=C_t, C_prev=C_prev, A_t=A_t,
                match_results=match_results,
                stop_state=stop_state,
                round_t=round_t,
            )
        else:
            snapshot = self.stopper.compute_signals(
                C_t, C_prev, A_t, match_results, round_t)
            stop_state.signal_history.append(snapshot)

        # STEP 9: Round metadata
        # model_checkpoint = source checkpoint used to generate C_t this round,
        # not the newly trained checkpoint (that is returned and used next round).
        # stop_condition_met = consecutive_satisfied > 0 means "counter is
        # accumulating" — set to stop_state.stopped to mean "pipeline stopped".
        last = stop_state.signal_history[-1]
        self.bank.write_metadata(RoundMetadata(
            round_id           = round_t,
            n_candidates       = sum(len(v) for v in C_t.values()),
            n_admitted         = len(A_t),
            raw_churn          = last.raw_churn,
            stable_yield       = last.stable_yield,
            class_drift        = last.class_drift,
            stop_condition_met = stop_state.consecutive_satisfied > 0,
            model_checkpoint   = checkpoint_t,  # source checkpoint for this round
        ))

        # STEP 10: Check stop
        if stop_state.stopped:
            return checkpoint_t, stop_state

        if round_t == self.config.max_rounds:
            stop_state.stopped     = True
            stop_state.stop_reason = "max_rounds"
            return checkpoint_t, stop_state

        # STEP 11: Retrain
        new_checkpoint = self.trainer.train(
            labeled_data=labeled_data,
            pseudo_labels=A_t,
            round_id=round_t,
        )
        return new_checkpoint, stop_state

    def _compute_jitter_ious(self,
                              model_t,
                              image_ids:     List[str],
                              images:        List[np.ndarray],
                              match_results: Dict[str, MatchResult],
                              idx_t:         CandidateIndex,
                              idx_prev:      CandidateIndex,
                              ) -> Dict[str, Tuple[float, float, float]]:
        """
        Frozen jitter selection rule (orchestrator owns entirely):
          For each matched pair where c_round_loc >= tau_pre:
            Run jitter_infer_fn for scale, translate, brightness.
            Only populate jitter_ious if all three passes succeed.
            If prefilter fails: omit from jitter_ious (never pass (0,0,0)).
        """
        images_by_id = dict(zip(image_ids, images))
        jitter_ious: Dict[str, Tuple[float, float, float]] = {}

        for mr in match_results.values():
            for box_id_t, box_id_prev in mr.matched_pairs:
                pl_t    = idx_t.get(box_id_t)
                pl_prev = idx_prev.get(box_id_prev)
                if pl_t is None or pl_prev is None:
                    continue

                c_round_loc = iou_pair(pl_t.box, pl_prev.box)
                if c_round_loc < self.config.scoring.tau_pre:
                    continue  # prefilter not passed — omit from jitter_ious

                image = images_by_id.get(pl_t.image_id)
                if image is None:
                    continue

                iou1 = self.jitter_infer_fn(
                    model_t, image, pl_t, "scale",      self.config)
                iou2 = self.jitter_infer_fn(
                    model_t, image, pl_t, "translate",  self.config)
                iou3 = self.jitter_infer_fn(
                    model_t, image, pl_t, "brightness", self.config)

                # Jitter convention (frozen): only populate if ALL three
                # passes return a valid IoU above min_iou_gate.
                # If any pass fails (returns 0.0), omit the box entirely —
                # never pass (0.0, 0.0, x) to scorer, because
                # std([0,0,0])=0 → C_jitter_loc=1.0 (wrong signal).
                ious = (iou1, iou2, iou3)
                if all(i >= self.config.matching.min_iou_gate for i in ious):
                    jitter_ious[box_id_t] = ious
                # Otherwise: omit → scorer sets C_jitter_loc=0.0, jitter_executed=False

        return jitter_ious

    def _write_bank_entries(self,
                             round_t:      int,
                             C_t:          CandidatePool,
                             match_results: Dict[str, MatchResult],
                             scores_t:     dict,
                             components_t: dict,
                             A_t:          list) -> None:
        """Write one BankEntry per image for this round."""
        admitted_ids = {pl.box_id for pl in A_t}
        all_images   = set(C_t.keys()) | set(match_results.keys())

        for image_id in all_images:
            pls  = C_t.get(image_id, [])
            mr   = match_results.get(image_id)
            if mr is None:
                continue

            prev_map = {bt: bp for bt, bp in mr.matched_pairs}

            self.bank.append(BankEntry(
                image_id            = image_id,
                round_id            = round_t,
                pseudo_labels       = pls,
                stability_scores    = {
                    pl.box_id: scores_t.get(pl.box_id) for pl in pls},
                admitted            = {
                    pl.box_id: pl.box_id in admitted_ids for pl in pls},
                matched_prev_box_id = {
                    pl.box_id: prev_map.get(pl.box_id) for pl in pls},
                is_ambiguous        = {
                    pl.box_id: pl.box_id in mr.ambiguous for pl in pls},
                c_cls_dist          = {
                    pl.box_id: components_t[pl.box_id].c_cls_dist
                    for pl in pls},
                c_round_loc         = {
                    pl.box_id: components_t[pl.box_id].c_round_loc
                    for pl in pls},
                c_jitter_loc        = {
                    pl.box_id: components_t[pl.box_id].c_jitter_loc
                    for pl in pls},
                jitter_executed     = {
                    pl.box_id: components_t[pl.box_id].jitter_executed
                    for pl in pls},
            ))