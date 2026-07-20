"""Sampler public exports.

Exports the functional ``diff_sample`` entry point, the corrector factory
``build_corrector``, and the clue injection utilities for biology prior
pre-filling at inference time.
"""

from __future__ import annotations

from .clue_injection import apply_clue_prefill, extract_clues, load_gene_id_name_mapping
from .default import diff_sample
from .utils.sampling_helpers import build_corrector

__all__ = [
    "apply_clue_prefill",
    "build_corrector",
    "diff_sample",
    "extract_clues",
    "load_gene_id_name_mapping",
]
