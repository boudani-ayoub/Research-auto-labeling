"""
utils/js_divergence.py
======================
Jensen-Shannon divergence, normalized to [0, 1] by dividing by ln(2).

Rules:
  - All JS calls in the system go through this module
  - Normalization by ln(2) is mandatory — raw JS is in [0, ln(2)]
  - Inputs are probability vectors that sum to 1.0
  - normalize() is the canonical pre-processing step before JS
  - If a vector sums to 0, it is replaced by a uniform distribution
  - CPU only
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a non-negative vector to sum to 1.0.
    If the vector sums to 0, returns a uniform distribution.
    This is the canonical pre-processing step before any JS call.
    """
    v   = np.asarray(v, dtype=np.float64)
    s   = v.sum()
    if s <= 0.0:
        return np.ones(len(v), dtype=np.float64) / len(v)
    return v / s


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two probability distributions,
    normalized to [0, 1] by dividing by ln(2).

    Both p and q must already sum to 1.0 (use normalize() first).
    If either is not already normalized, normalization is applied silently.

    Returns a value in [0, 1]:
        0.0 — identical distributions
        1.0 — maximally different distributions

    Formula:
        M   = 0.5 * (p + q)
        JS  = 0.5 * KL(p||M) + 0.5 * KL(q||M)
        JSD = JS / ln(2)     (normalized to [0,1])

    Uses the convention 0 * log(0) = 0.
    """
    p = normalize(np.asarray(p, dtype=np.float64))
    q = normalize(np.asarray(q, dtype=np.float64))

    if len(p) != len(q):
        raise ValueError(
            f"js_divergence: p and q must have the same length, "
            f"got {len(p)} and {len(q)}"
        )

    m = 0.5 * (p + q)

    # KL(p||m) with 0*log(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0.0, p * np.log(p / np.where(m > 0.0, m, 1.0)), 0.0)
        kl_qm = np.where(q > 0.0, q * np.log(q / np.where(m > 0.0, m, 1.0)), 0.0)

    js_raw      = 0.5 * kl_pm.sum() + 0.5 * kl_qm.sum()
    js_raw      = float(np.clip(js_raw, 0.0, np.log(2)))  # numerical safety
    js_normalized = js_raw / np.log(2)

    return float(np.clip(js_normalized, 0.0, 1.0))


def js_divergence_marginals(class_counts_t: np.ndarray,
                             class_counts_prev: np.ndarray) -> float:
    """
    JS divergence between two class-count vectors (unnormalized histograms).
    Normalizes both to probability distributions before computing JS.
    Used for ClassDrift in the stopping evaluator.

    Returns value in [0, 1].
    """
    p = normalize(np.asarray(class_counts_t,    dtype=np.float64))
    q = normalize(np.asarray(class_counts_prev, dtype=np.float64))
    return js_divergence(p, q)