"""
bank/bank.py
============
Persistent pseudo-label bank.

Design constraints from architecture spec:
  - Append-only per round — never overwrite previous round entries
  - Fully resumable from disk (load_or_create pattern)
  - Exposes: get_candidate_pool(round_id), append(BankEntry),
             write_metadata(RoundMetadata)
  - Matching and scoring run on CPU — bank is CPU-only storage
  - No computation here — pure storage and retrieval

Storage format:
  One JSON file per round: {bank_dir}/round_{round_id:04d}.json
  One metadata file:       {bank_dir}/metadata.json

  Splitting by round means:
    - get_candidate_pool(round_id) reads exactly one file
    - Append never touches previous round files
    - Resumability is trivial: scan for existing round files on load
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from autolabel.bank.schemas import (
    BankEntry,
    CandidatePool,
    PseudoLabel,
    RoundMetadata,
)


# ── Serialization helpers ─────────────────────────────────────────────────────

def _pseudo_label_to_dict(pl: PseudoLabel) -> dict:
    return {
        "image_id":     pl.image_id,
        "box_id":       pl.box_id,
        "round_id":     pl.round_id,
        "box":          list(pl.box),
        "pred_class":   pl.pred_class,
        "class_scores": list(pl.class_scores),
        "confidence":   pl.confidence,
    }


def _pseudo_label_from_dict(d: dict) -> PseudoLabel:
    return PseudoLabel(
        image_id    = d["image_id"],
        box_id      = d["box_id"],
        round_id    = d["round_id"],
        box         = tuple(d["box"]),
        pred_class  = d["pred_class"],
        class_scores= tuple(d["class_scores"]),
        confidence  = d["confidence"],
    )


def _bank_entry_to_dict(entry: BankEntry) -> dict:
    return {
        "image_id":            entry.image_id,
        "round_id":            entry.round_id,
        "pseudo_labels":       [_pseudo_label_to_dict(pl)
                                for pl in entry.pseudo_labels],
        "stability_scores":    entry.stability_scores,
        "admitted":            entry.admitted,
        "matched_prev_box_id": entry.matched_prev_box_id,
        "is_ambiguous":        entry.is_ambiguous,
        "c_cls_dist":          entry.c_cls_dist,
        "c_round_loc":         entry.c_round_loc,
        "c_jitter_loc":        entry.c_jitter_loc,
        "jitter_executed":     entry.jitter_executed,
    }


def _bank_entry_from_dict(d: dict) -> BankEntry:
    return BankEntry(
        image_id            = d["image_id"],
        round_id            = d["round_id"],
        pseudo_labels       = [_pseudo_label_from_dict(pl)
                               for pl in d["pseudo_labels"]],
        stability_scores    = {k: v for k, v in d["stability_scores"].items()},
        admitted            = {k: bool(v) for k, v in d["admitted"].items()},
        matched_prev_box_id = {k: v for k, v in d["matched_prev_box_id"].items()},
        is_ambiguous        = {k: bool(v) for k, v in d["is_ambiguous"].items()},
        c_cls_dist          = {k: v for k, v in d["c_cls_dist"].items()},
        c_round_loc         = {k: v for k, v in d["c_round_loc"].items()},
        c_jitter_loc        = {k: v for k, v in d["c_jitter_loc"].items()},
        jitter_executed     = {k: bool(v) for k, v in d["jitter_executed"].items()},
    )


def _round_metadata_to_dict(meta: RoundMetadata) -> dict:
    return {
        "round_id":           meta.round_id,
        "n_candidates":       meta.n_candidates,
        "n_admitted":         meta.n_admitted,
        "raw_churn":          meta.raw_churn,
        "stable_yield":       meta.stable_yield,
        "class_drift":        meta.class_drift,
        "stop_condition_met": meta.stop_condition_met,
        "model_checkpoint":   meta.model_checkpoint,
    }


def _round_metadata_from_dict(d: dict) -> RoundMetadata:
    return RoundMetadata(
        round_id           = d["round_id"],
        n_candidates       = d["n_candidates"],
        n_admitted         = d["n_admitted"],
        raw_churn          = d["raw_churn"],
        stable_yield       = d["stable_yield"],
        class_drift        = d["class_drift"],
        stop_condition_met = bool(d["stop_condition_met"]),
        model_checkpoint   = d["model_checkpoint"],
    )


# ── Bank ──────────────────────────────────────────────────────────────────────

class PseudoLabelBank:
    """
    Persistent append-only storage for pseudo-label entries and round metadata.

    Directory layout:
        {bank_dir}/
            round_0000.json   — BankEntry list for round 0
            round_0001.json   — BankEntry list for round 1
            ...
            metadata.json     — list of RoundMetadata, one per round written

    Usage:
        bank = PseudoLabelBank.load_or_create("outputs/bank")
        bank.append(entry)            # called once per image per round
        bank.write_metadata(meta)     # called once per round
        pool = bank.get_candidate_pool(round_id=2)
    """

    METADATA_FILENAME = "metadata.json"

    def __init__(self, bank_dir: str):
        self._bank_dir  = Path(bank_dir)
        self._bank_dir.mkdir(parents=True, exist_ok=True)
        # In-memory write buffer for current round — flushed on write_metadata
        self._write_buffer: List[BankEntry]       = []
        self._current_round: Optional[int]        = None
        self._metadata: List[RoundMetadata]       = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, bank_dir: str) -> "PseudoLabelBank":
        """
        Load an existing bank from disk, or create a new empty one.
        Resumability: scans for existing round files and metadata on disk.
        """
        bank = cls(bank_dir)
        bank._load_metadata()
        return bank

    # ── Write API ─────────────────────────────────────────────────────────────

    def append(self, entry: BankEntry) -> None:
        """
        Buffer a BankEntry for the current round.
        Entries are written to disk when write_metadata() is called.
        All entries in one write_metadata() call must share the same round_id.
        """
        if self._current_round is None:
            self._current_round = entry.round_id
        elif entry.round_id != self._current_round:
            raise ValueError(
                f"BankEntry round_id={entry.round_id} does not match "
                f"current round {self._current_round}. "
                f"Call write_metadata() to close the current round first."
            )
        self._write_buffer.append(entry)

    def write_metadata(self, meta: RoundMetadata) -> None:
        """
        Flush the write buffer for the current round to disk and record metadata.
        After this call the round is closed — no more entries can be appended
        for that round_id.

        If no entries were buffered (e.g. round 0 burn-in), writes an empty
        round file so round_id is still retrievable.

        Raises ValueError if meta.round_id does not match the round_id of the
        buffered entries — prevents silently writing round-1 entries into a
        round-2 file.
        """
        round_id = meta.round_id

        # Consistency check: buffered entries must match meta.round_id
        if self._current_round is not None and self._current_round != round_id:
            raise ValueError(
                f"write_metadata called with round_id={round_id} but "
                f"buffered entries belong to round {self._current_round}. "
                f"Call write_metadata(round_id={self._current_round}) to "
                f"close the current round first."
            )

        round_path = self._round_path(round_id)

        if round_path.exists():
            raise FileExistsError(
                f"Round file already exists: {round_path}. "
                f"Bank is append-only — cannot overwrite a committed round."
            )

        # Write round entries
        entries_dicts = [_bank_entry_to_dict(e) for e in self._write_buffer]
        with open(round_path, "w") as f:
            json.dump(entries_dicts, f, indent=2)

        # Write metadata
        self._metadata.append(meta)
        self._flush_metadata()

        # Reset buffer
        self._write_buffer  = []
        self._current_round = None

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_candidate_pool(self, round_id: int) -> CandidatePool:
        """
        Return CandidatePool for the given round_id.
        Reads exactly one file from disk.
        Returns empty pool if round has no entries (e.g. round 0 burn-in).

        Raises FileNotFoundError if round_id has never been written.
        """
        round_path = self._round_path(round_id)
        if not round_path.exists():
            raise FileNotFoundError(
                f"No bank file for round_id={round_id} at {round_path}. "
                f"Available rounds: {self.committed_rounds()}"
            )

        with open(round_path, "r") as f:
            entries_dicts = json.load(f)

        pool: CandidatePool = {}
        for d in entries_dicts:
            entry = _bank_entry_from_dict(d)
            pool[entry.image_id] = entry.pseudo_labels

        return pool

    def get_metadata(self, round_id: int) -> Optional[RoundMetadata]:
        """Return RoundMetadata for round_id, or None if not found."""
        for meta in self._metadata:
            if meta.round_id == round_id:
                return meta
        return None

    def all_metadata(self) -> List[RoundMetadata]:
        """Return all RoundMetadata records in round order."""
        return sorted(self._metadata, key=lambda m: m.round_id)

    def committed_rounds(self) -> List[int]:
        """Return sorted list of round_ids that have been written to disk."""
        return sorted(m.round_id for m in self._metadata)

    def latest_round_id(self) -> Optional[int]:
        """Return the highest committed round_id, or None if bank is empty."""
        rounds = self.committed_rounds()
        return rounds[-1] if rounds else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _round_path(self, round_id: int) -> Path:
        return self._bank_dir / f"round_{round_id:04d}.json"

    def _metadata_path(self) -> Path:
        return self._bank_dir / self.METADATA_FILENAME

    def _flush_metadata(self) -> None:
        """Write all metadata records to disk atomically via temp file."""
        meta_path = self._metadata_path()
        tmp_path  = meta_path.with_suffix(".tmp")
        data      = [_round_metadata_to_dict(m) for m in self._metadata]
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, meta_path)  # atomic on POSIX

    def _load_metadata(self) -> None:
        """Load metadata from disk if it exists (resumability)."""
        meta_path = self._metadata_path()
        if meta_path.exists():
            with open(meta_path, "r") as f:
                data = json.load(f)
            self._metadata = [_round_metadata_from_dict(d) for d in data]
        else:
            self._metadata = []