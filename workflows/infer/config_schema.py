"""Structured config schema for the inference workflow.

Mirrors ``configs/infer/default.yaml`` so Hydra can validate the
composed configuration at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.workflow_utils import choice_field


@dataclass
class OutputConfig:
    """Inference output directory layout.

    Attributes:
        workdir: Base working directory for all inference outputs.
        sample_dir: Sub-directory name for this sampling run. The YAML
            default uses Hydra interpolation to auto-generate a name
            from checkpoint and sampler parameters. Can be overridden
            via CLI (e.g. ``output.sample_dir=my_run``).
    """

    workdir: str = MISSING
    sample_dir: str = ""


@dataclass
class CheckpointConfig:
    """How to locate the model checkpoint for inference.

    Two mutually exclusive modes:

    **New-format (is_legacy=false)**: Use ``training_dir`` + ``iteration`` +
    ``ema_std`` to auto-discover a checkpoint at
    ``{training_dir}/ema/step-{iteration}-ema-*samples-{ema_std}std.pth``.
    The training directory's ``.hydra/config.yaml`` is also loaded to
    override tokenizer/model configs.

    **Legacy (is_legacy=true)**: Use ``model_path`` directly. ``iteration``
    and ``ema_std`` are auto-parsed from the legacy filename pattern
    ``iter-XXXXXX-ema-XXXXXXsamples-X.XXXstd.pth``. Tokenizer/model must
    be specified separately (e.g. via a legacy-specific infer config).

    Attributes:
        is_legacy: Whether the checkpoint is in the legacy (old repo) format.
        training_dir: New-format training directory (ignored when is_legacy=true).
        model_path: Explicit checkpoint file path. Required when is_legacy=true.
        iteration: Training iteration (ignored when is_legacy=true, auto-parsed).
        ema_std: EMA standard deviation (ignored when is_legacy=true, auto-parsed).
    """

    is_legacy: bool = False
    training_dir: str = ""
    model_path: str = ""
    iteration: int = 0
    ema_std: float = 0.050


@dataclass
class TaskConfig:
    """Per-task generation settings.

    Attributes:
        subtask: Subtask specification ``"3of16"`` for distributed processing.
        base_seed: Starting random seed.
        save_every_n_samples: Periodic save interval for resume-friendliness.
        condition_csv: Generation conditions as either a file path to a CSV
            or inline CSV content (multi-line string). For Hydra CLI,
            wrap in single quotes to escape commas:
            ``"task.condition_csv='tissue,n_cells\nheart,1000'"``.
            Column names should match the tokenizer's ConditionFormat
            (e.g. ``tissue``, ``cell_type``).
        batch_size: Batch size for sampling.
        obs_passthrough_columns: Extra condition-CSV columns to copy verbatim
            into the generated h5ad ``obs`` (in addition to the ConditionFormat
            slot columns). Use for labels the downstream eval needs but the
            model does not condition on — e.g. ``cell_type_coarse`` for the
            batch-effect cLISI, or ``cell_line``/``drug``/dose for Tahoe
            cell-eval grouping. Empty (default) = only slot columns reach obs.
        filename_columns: Condition-CSV columns used to build the per-task
            output filename prefix. When non-empty, ONLY these (short)
            identifier columns are used (no truncation) — e.g.
            ``[cell_line, drug, drug_concentration_with_unit]``. Empty
            (default) = legacy behaviour (all condition columns, long values
            truncated + hashed).
    """

    subtask: str = "1of1"
    base_seed: int = 0
    save_every_n_samples: int = 50
    condition_csv: str = ""
    batch_size: int = 1
    obs_passthrough_columns: List[str] = field(default_factory=list)
    filename_columns: List[str] = field(default_factory=list)


@dataclass
class InferConfig:
    """Top-level inference workflow configuration."""

    name: str = MISSING
    seed: int = MISSING
    device: str = choice_field("auto", "cpu", "cuda", "mps", default=MISSING)
    dtype: str = choice_field("auto", "float32", "float16", "bfloat16", default=MISSING)
    output: OutputConfig = field(default_factory=OutputConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    # Hydra-instantiated modules (opaque dicts at schema level).
    tokenizer: dict[str, Any] = MISSING
    model: dict[str, Any] = MISSING
    sampler: dict[str, Any] = MISSING


ConfigStore.instance().store(name="infer_schema", node=InferConfig)


__all__ = [
    "CheckpointConfig",
    "InferConfig",
    "OutputConfig",
    "TaskConfig",
]
