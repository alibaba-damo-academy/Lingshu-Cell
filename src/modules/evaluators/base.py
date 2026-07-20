"""Base protocol for evaluation metric modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anndata as ad


@dataclass
class ExternalEvalParams:
    """Base parameters shared by all external evaluators.

    Subclass this for evaluator-specific parameters. The workflow
    instantiates the subclass from the ``params`` sub-dict in YAML,
    providing type validation and centralized defaults.

    Note: ``predict_filename`` is a workflow-level routing field (which
    file to evaluate in sample_dir) and does NOT belong here. It lives
    in the outer YAML entry, handled by the workflow before constructing
    this object.
    """

    pass


@runtime_checkable
class MetricModule(Protocol):
    """Protocol for standard evaluation metric modules.

    Standard metrics receive preprocessed AnnData objects (already
    normalized, gene-aligned, and sampled) and compute scalar metrics
    and/or save visualizations to ``output_dir``.
    """

    name: str

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the metric.

        Args:
            gen_adata: Preprocessed generated data.
            real_adata: Preprocessed real data.
            output_dir: Directory to save outputs (PNG, CSV, etc.).
            device: ``"cpu"`` or ``"cuda"``.
            **kwargs: Metric-specific parameters from config.

        Returns:
            Dict of scalar metric results (e.g. ``{"pearson": 0.95}``).
        """
        ...


@runtime_checkable
class ExternalEvalModule(Protocol):
    """Protocol for external evaluation tools (shell scripts, etc.).

    External modules receive the raw merged h5ad path and handle
    their own data loading independently.

    Implementations must declare:
        - ``params_class``: class attribute pointing to the parameter
          dataclass (a subclass of ``ExternalEvalParams``).
        - ``_external = True``: marker for workflow dispatch.
    """

    name: str
    params_class: type[ExternalEvalParams]

    def compute(
        self,
        *,
        gen_h5ad_path: Path,
        output_dir: Path,
        params: ExternalEvalParams,
    ) -> dict[str, Any]:
        """Run external evaluation.

        Args:
            gen_h5ad_path: Path to the merged generated h5ad file.
            output_dir: Directory to save outputs.
            params: Validated parameter object (subclass of ExternalEvalParams).

        Returns:
            Dict of scalar metric results parsed from tool output.
        """
        ...
