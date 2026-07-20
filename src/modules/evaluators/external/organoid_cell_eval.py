"""Cell-eval wrapper for per-cell-type organoid perturbation evaluation."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .. import register_metric
from ..base import ExternalEvalParams


SCRIPT_PATH = Path(__file__).parent / "scripts" / "Norman2019_Double_Gene_Eval.sh"
PREDICT_FILENAME = "predict_with_control.h5ad"
_ACTIVATE_AND_RUN = 'source activate "$1" && shift && exec bash "$@"'


@dataclass
class OrganoidCellEvalParams(ExternalEvalParams):
    """Parameters for cell-type-specific organoid cell-eval runs.

    Attributes:
        cell_types: Cell types to evaluate with equal macro weight.
        eval_meta_json: JSON path containing per-cell-type reference paths.
        conda_env: Conda environment containing the ``cell-eval`` executable.
        threads: Number of threads passed to ``cell-eval run``.
        profile: Cell-eval profile name.
        pert_col: Observation column containing perturbation labels.
        control: Control perturbation label.
    """

    cell_types: list[str] = field(default_factory=list)
    eval_meta_json: str = ""
    conda_env: str = ""
    threads: int = 20
    profile: str = "vcc_general"
    pert_col: str = "target_gene"
    control: str = "non-targeting"


def _resolve_existing_path(value: str | Path, label: str) -> Path:
    """Resolve a path relative to the current project directory.

    Args:
        value: Absolute or project-relative path.
        label: Human-readable path label for errors.

    Returns:
        Absolute resolved path.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validated_cell_types(params: OrganoidCellEvalParams) -> tuple[str, ...]:
    """Validate and normalize configured cell types.

    Args:
        params: Evaluator parameters to validate.

    Returns:
        Cell types in configured order.

    Raises:
        ValueError: If parameters are empty, duplicated, or unsafe.
    """
    if isinstance(params.cell_types, (str, bytes)) or not isinstance(
        params.cell_types, Sequence
    ):
        raise ValueError("cell_types must be a list of cell-type names")
    cell_types = tuple(str(cell_type) for cell_type in params.cell_types)
    if not cell_types:
        raise ValueError("cell_types must contain at least one cell type")
    if len(set(cell_types)) != len(cell_types):
        raise ValueError(f"cell_types contains duplicates: {cell_types}")
    for cell_type in cell_types:
        if not cell_type or Path(cell_type).name != cell_type or cell_type in {".", ".."}:
            raise ValueError(f"Invalid cell type path component: {cell_type!r}")

    string_params = {
        "eval_meta_json": params.eval_meta_json,
        "conda_env": params.conda_env,
        "profile": params.profile,
        "pert_col": params.pert_col,
        "control": params.control,
    }
    empty = [name for name, value in string_params.items() if not str(value).strip()]
    if empty:
        raise ValueError(f"Parameters must be non-empty: {empty}")
    if isinstance(params.threads, bool) or not isinstance(params.threads, int):
        raise ValueError("threads must be an integer")
    if params.threads <= 0:
        raise ValueError("threads must be positive")
    return cell_types


def _load_reference_paths(
    params: OrganoidCellEvalParams,
    cell_types: Sequence[str],
) -> dict[str, dict[str, Path]]:
    """Load and validate per-cell-type paths from the metadata JSON.

    The JSON may contain the cell-type mapping directly or under a
    ``cell_types`` key. Each selected entry must define ``real_h5ad``,
    ``real_de``, and ``baseline_agg``.

    Args:
        params: Evaluator parameters containing the metadata JSON path.
        cell_types: Selected cell types in evaluation order.

    Returns:
        Cell types mapped to absolute reference paths.

    Raises:
        TypeError: If the JSON or a cell-type entry is not an object.
        ValueError: If a selected cell type or required field is missing.
        FileNotFoundError: If the metadata or a referenced file is missing.
    """
    meta_path = _resolve_existing_path(params.eval_meta_json, "eval metadata JSON")
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, Mapping):
        raise TypeError("eval metadata JSON must contain an object")

    entries = meta.get("cell_types", meta)
    if not isinstance(entries, Mapping):
        raise TypeError("eval metadata 'cell_types' must contain an object")

    references: dict[str, dict[str, Path]] = {}
    required_fields = ("real_h5ad", "real_de", "baseline_agg")
    for cell_type in cell_types:
        if cell_type not in entries:
            raise ValueError(
                f"Cell type {cell_type!r} is absent from eval metadata; "
                f"available: {list(entries)}"
            )
        entry = entries[cell_type]
        if not isinstance(entry, Mapping):
            raise TypeError(f"Metadata for cell type {cell_type!r} must be an object")
        missing = [field_name for field_name in required_fields if not entry.get(field_name)]
        if missing:
            raise ValueError(
                f"Metadata for cell type {cell_type!r} is missing fields: {missing}"
            )
        references[cell_type] = {
            field_name: _resolve_existing_path(
                str(entry[field_name]), f"{cell_type} {field_name}"
            )
            for field_name in required_fields
        }
    return references


def _build_command(
    *,
    predict_path: Path,
    output_dir: Path,
    references: Mapping[str, Path],
    params: OrganoidCellEvalParams,
) -> list[str]:
    """Build a shell-safe argv list for one cell-eval script invocation.

    The only shell program is a fixed activation expression. All configurable
    values are positional arguments and are never interpolated into it.

    Args:
        predict_path: Generated H5AD containing control cells.
        output_dir: Cell-type-specific result directory.
        references: Real H5AD, DE CSV, and baseline aggregate paths.
        params: Cell-eval runtime parameters.

    Returns:
        Argument vector suitable for ``subprocess.run``.
    """
    script_args = [
        str(SCRIPT_PATH.resolve()),
        "--predict",
        str(predict_path),
        "--validation",
        str(references["real_h5ad"]),
        "--real-de",
        str(references["real_de"]),
        "--base-agg",
        str(references["baseline_agg"]),
        "--outdir",
        str(output_dir),
        "--threads",
        str(params.threads),
        "--profile",
        params.profile,
        "--pert-col",
        params.pert_col,
        "--control-pert",
        params.control,
    ]
    return [
        "bash",
        "-c",
        _ACTIVATE_AND_RUN,
        "organoid-cell-eval",
        params.conda_env,
        *script_args,
    ]


def _run_cell_eval(command: Sequence[str], cell_type: str) -> None:
    """Run one external cell-eval command and surface useful failures.

    Args:
        command: Complete subprocess argument vector.
        cell_type: Cell type being evaluated for error context.

    Raises:
        RuntimeError: If the external script exits unsuccessfully.
    """
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-2000:] if result.stderr else "<empty stderr>"
        raise RuntimeError(
            f"Organoid cell-eval failed for {cell_type!r} "
            f"(exit {result.returncode}):\n{stderr}"
        )


def _numeric_value(value: Any) -> float | None:
    """Convert a scalar to a finite float when possible.

    Args:
        value: CSV scalar value.

    Returns:
        Finite float value, or ``None`` for missing/non-numeric values.
    """
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_results(output_dir: Path) -> dict[str, float]:
    """Parse aggregate means and baseline-difference scores.

    Args:
        output_dir: Cell-type-specific cell-eval output directory.

    Returns:
        Flat numeric metrics prefixed with ``agg_`` or ``score_``.

    Raises:
        FileNotFoundError: If aggregate or score output is absent.
        ValueError: If an output CSV has an unexpected schema or no metrics.
    """
    agg_path = output_dir / "agg_results.csv"
    if not agg_path.exists():
        candidates = sorted(output_dir.rglob("*agg_results.csv"))
        if not candidates:
            raise FileNotFoundError(f"agg_results.csv not found under {output_dir}")
        agg_path = candidates[0]

    agg = pd.read_csv(agg_path)
    if "statistic" not in agg.columns:
        raise ValueError(f"Aggregate output lacks 'statistic': {agg_path}")
    mean_rows = agg[agg["statistic"].astype(str).str.lower() == "mean"]
    if mean_rows.empty:
        raise ValueError(f"Aggregate output lacks a mean row: {agg_path}")

    metrics: dict[str, float] = {}
    mean_row = mean_rows.iloc[0]
    for column in agg.columns:
        if column == "statistic":
            continue
        value = _numeric_value(mean_row[column])
        if value is not None:
            metrics[f"agg_{column}"] = value

    score_path = output_dir / "baseline_diff.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"baseline_diff.csv not found: {score_path}")
    scores = pd.read_csv(score_path)
    required_columns = {"metric", "from_baseline"}
    if not required_columns.issubset(scores.columns):
        raise ValueError(
            f"Baseline score output lacks columns {sorted(required_columns)}: {score_path}"
        )
    for row in scores.itertuples(index=False):
        metric_name = str(getattr(row, "metric"))
        value = _numeric_value(getattr(row, "from_baseline"))
        if value is not None:
            metrics[f"score_{metric_name}"] = value

    if not metrics:
        raise ValueError(f"No numeric metrics found under {output_dir}")
    return metrics


def _macro_metrics(
    cell_type_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Average shared numeric metrics with equal cell-type weights.

    Args:
        cell_type_metrics: Per-cell-type parsed metrics.

    Returns:
        Metrics prefixed with ``macro_``.

    Raises:
        ValueError: If cell types have no shared numeric metric.
    """
    iterator = iter(cell_type_metrics.values())
    metric_names = set(next(iterator))
    for metrics in iterator:
        metric_names.intersection_update(metrics)
    if not metric_names:
        raise ValueError("Cell types have no shared numeric metric for macro averaging")
    return {
        f"macro_{metric_name}": sum(
            metrics[metric_name] for metrics in cell_type_metrics.values()
        )
        / len(cell_type_metrics)
        for metric_name in sorted(metric_names)
    }


@register_metric("organoid_cell_eval")
class OrganoidCellEvalMetric:
    """Run Norman2019 cell-eval separately for each organoid cell type."""

    name = "organoid_cell_eval"
    _external = True
    params_class = OrganoidCellEvalParams

    def compute(
        self,
        *,
        gen_h5ad_path: Path,
        output_dir: Path,
        params: OrganoidCellEvalParams,
    ) -> dict[str, float]:
        """Run per-cell-type cell-eval and write macro summaries.

        Args:
            gen_h5ad_path: Input directory containing one subdirectory per
                cell type, each with ``predict_with_control.h5ad``.
            output_dir: Root directory for per-cell-type outputs and summaries.
            params: Organoid cell-eval parameters.

        Returns:
            Cell-type-prefixed metrics plus equal-weight ``macro_`` metrics.

        Raises:
            TypeError: If ``params`` has the wrong dataclass type.
            FileNotFoundError: If an input, reference, script, or result is absent.
            ValueError: If parameters, metadata, or result schemas are invalid.
            RuntimeError: If an external evaluation invocation fails.
        """
        if not isinstance(params, OrganoidCellEvalParams):
            raise TypeError(
                "organoid_cell_eval params must be OrganoidCellEvalParams, got "
                f"{type(params).__name__}"
            )
        cell_types = _validated_cell_types(params)
        if not SCRIPT_PATH.is_file():
            raise FileNotFoundError(f"Cell-eval script not found: {SCRIPT_PATH}")
        input_dir = _resolve_existing_path(gen_h5ad_path, "cell-eval input directory")
        if not input_dir.is_dir():
            raise ValueError(f"cell-eval input path is not a directory: {input_dir}")
        references = _load_reference_paths(params, cell_types)

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cell_type_metrics: dict[str, dict[str, float]] = {}
        for cell_type in cell_types:
            predict_path = _resolve_existing_path(
                input_dir / cell_type / PREDICT_FILENAME,
                f"{cell_type} generated H5AD",
            )
            cell_output_dir = output_dir / cell_type
            cell_output_dir.mkdir(parents=True, exist_ok=True)
            command = _build_command(
                predict_path=predict_path,
                output_dir=cell_output_dir,
                references=references[cell_type],
                params=params,
            )
            print(f"  Running organoid cell-eval: {cell_type}")
            _run_cell_eval(command, cell_type)
            cell_type_metrics[cell_type] = _parse_results(cell_output_dir)

        macro = _macro_metrics(cell_type_metrics)
        with (output_dir / "cell_type_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(cell_type_metrics, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        pd.DataFrame(
            [{"metric": metric_name, "value": value} for metric_name, value in macro.items()]
        ).to_csv(output_dir / "macro_summary.csv", index=False, lineterminator="\n")

        results = {
            f"{cell_type}_{metric_name}": value
            for cell_type, metrics in cell_type_metrics.items()
            for metric_name, value in metrics.items()
        }
        results.update(macro)
        return results


__all__ = ["OrganoidCellEvalMetric", "OrganoidCellEvalParams"]
