"""Logit and token corrector exports."""

from __future__ import annotations

from .dense import DenseLogitCorrector
from .sparse import SparseLogitCorrector

__all__ = ["DenseLogitCorrector", "SparseLogitCorrector"]
