"""Structured config schema for the eval workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING



@dataclass
class EvalConfig:
    """Top-level eval workflow configuration.

    Attributes:
        name: Run name.
        device: Compute device (``auto``, ``cpu``, ``cuda``).
        sample_dir: Directory containing generated h5ad (infer output).
        real_data: Path to real data h5ad file.
        gene_id_name_file: Optional Ensembl ID → gene name mapping file.
        convert_ensembl_to_name: Whether to convert real data gene IDs.
        num_samples: Number of cells to sample from each dataset for eval.
        seed: Random seed for sampling.
        metrics: List of metric names to compute.
        external: Per-metric-name config for external eval tools. Keys
            should match metric names in ``metrics`` list (e.g. ``vcc_train``,
            ``parse``).
    """

    name: str = "eval"
    device: str = "auto"
    sample_dir: str = MISSING
    real_data: str = ""
    gene_id_name_file: str = ""
    convert_ensembl_to_name: bool = False
    num_samples: int = 10000
    seed: int = 42
    metrics: List[str] = field(default_factory=lambda: ["umap", "distribution", "linregress"])
    external: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(name="eval_schema", node=EvalConfig)


__all__ = ["EvalConfig"]
