"""VCC/PARSE/Tahoe external evaluation tool wrapper.

Calls external shell scripts (VCC_Score_general.sh, Tahoe_Drug_Eval.sh, etc.)
under a specified conda environment and parses their output (agg_results.csv).

Available scripts (bundled in this directory):
- VCC_Score_general.sh: VCC train-split evaluation
- VCC_Score_general_VCC_Val.sh: VCC validation set
- VCC_Score_general_VCC_Test.sh: VCC test set
- PARSE_Organoid_Simu_script.sh: PARSE perturbation response
- Tahoe_Drug_Eval.sh: Tahoe drug perturbation (per split × cell_line)

All scripts require the ``cell-eval`` CLI tool, available in the
``vcc_general`` conda environment.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import register_metric
from ..base import ExternalEvalParams

SCRIPTS_DIR = Path(__file__).parent / "scripts"


@dataclass
class VCCScoreParams(ExternalEvalParams):
    """Parameters for VCC/PARSE external evaluation.

    Attributes:
        script_name: Shell script filename in ``scripts/`` directory.
        conda_env: Conda environment path containing ``cell-eval`` CLI.
        threads: Number of threads for cell-eval.
        exp_id: PARSE experiment ID (only for PARSE scripts).
        donor_id: PARSE donor ID (only for PARSE scripts).
    """

    script_name: str = "VCC_Score_general.sh"
    conda_env: str = ""
    threads: int = 40
    exp_id: str = ""
    donor_id: str = ""


@dataclass
class TahoeDrugEvalParams(ExternalEvalParams):
    """Parameters for Tahoe drug perturbation external evaluation.

    Resolves validation h5ad, real_de CSV, and baseline agg CSV paths
    from a meta JSON file indexed by (split, cell_line).

    Attributes:
        script_name: Shell script filename in ``scripts/`` directory.
        conda_env: Conda environment path containing ``cell-eval`` CLI.
        threads: Number of threads for cell-eval.
        eval_meta_json: Path to the meta JSON containing per-split/cell_line paths.
        split: Split name (e.g. ``val_uc_pair``, ``val_uc_dose``, ``val_ud``).
        cell_line: Cell line identifier (e.g. ``CVCL_0399``).
        profile: cell-eval profile name.
        pert_col: Perturbation column name in h5ad obs.
        control_pert: Control perturbation identifier.
        celltype_col: Cell type column name in h5ad obs.
    """

    script_name: str = "Tahoe_Drug_Eval.sh"
    conda_env: str = ""
    threads: int = 40
    eval_meta_json: str = ""
    split: str = ""
    cell_line: str = ""
    profile: str = "vcc_general"
    pert_col: str = "drug__dose"
    control_pert: str = "DMSO_TF__0.0 uM"
    celltype_col: str = "cell_line_id"


@dataclass
class Norman2019EvalParams(ExternalEvalParams):
    """Parameters for Norman2019 double-gene perturbation external evaluation.

    Unlike Tahoe, paths are specified directly (no meta JSON indirection)
    since there is a single cell line (K562) and eval sets are parameterized
    via ``eval_set``.

    Attributes:
        script_name: Shell script filename in ``scripts/`` directory.
        conda_env: Conda environment path containing ``cell-eval`` CLI.
        threads: Number of threads for cell-eval.
        real_h5ad: Path to real validation h5ad (control + perturbed cells).
        real_de_csv: Path to real DE CSV (from cell-eval baseline).
        baseline_agg_csv: Path to baseline agg_results.csv
            (from cell-eval run on baseline predictions).
        eval_set: Evaluation set name (``test_double`` or ``sanity_single``).
        profile: cell-eval profile name.
        pert_col: Perturbation column name in h5ad obs.
        control_pert: Control perturbation identifier.
    """

    script_name: str = "Norman2019_Double_Gene_Eval.sh"
    conda_env: str = ""
    threads: int = 20
    real_h5ad: str = ""
    real_de_csv: str = ""
    baseline_agg_csv: str = ""
    eval_set: str = "test_double"
    profile: str = "vcc_general"
    pert_col: str = "target_gene"
    control_pert: str = "non-perturbation"


class VCCScoreMetric:
    """External VCC/PARSE evaluation via shell script.

    Runs a bundled eval shell script under a specified conda environment,
    then parses the output ``agg_results.csv``.

    Multiple instances can be registered under different names (e.g.
    ``vcc_train``, ``vcc_val``, ``parse``) with different configs.
    """

    _external = True
    params_class = VCCScoreParams

    def __init__(self, name: str = "vcc_score"):
        self.name = name

    def compute(
        self,
        *,
        gen_h5ad_path: Path,
        output_dir: Path,
        params: VCCScoreParams,
    ) -> dict[str, Any]:
        """Run external VCC/PARSE evaluation script.

        Args:
            gen_h5ad_path: Path to the merged generated h5ad.
            output_dir: Directory for evaluation outputs.
            params: Validated parameter object.

        Returns:
            Dict of parsed metric results from agg_results.csv.
        """
        script_name = params.script_name
        conda_env = params.conda_env
        threads = params.threads

        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(
                f"External eval script not found: {script_path}\n"
                f"Available: {[f.name for f in SCRIPTS_DIR.glob('*.sh')]}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve to absolute paths since the subprocess runs with cwd=SCRIPTS_DIR.
        abs_predict = Path(gen_h5ad_path).resolve()
        abs_outdir = output_dir.resolve()

        # Build command.
        cmd = [
            "bash", "-c",
            f"source activate {conda_env} && "
            f"bash {script_path} "
            f"--predict {abs_predict} "
            f"--outdir {abs_outdir} "
            f"--threads {threads}"
            + _build_extra_args(params),
        ]

        print(f"  Running: {script_name}")
        print(f"  Env: {conda_env}")
        print(f"  Predict: {abs_predict}")
        print(f"  Output: {abs_outdir}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR),
        )

        if result.returncode != 0:
            print(f"  STDERR:\n{result.stderr[-2000:]}")
            raise RuntimeError(
                f"External eval script failed (exit {result.returncode}): {script_name}"
            )

        # Parse results.
        return _parse_results(abs_outdir)


def _build_extra_args(params: ExternalEvalParams) -> str:
    """Build extra CLI arguments based on params type.

    Args:
        params: Parameter object (VCCScoreParams, TahoeDrugEvalParams,
            or Norman2019EvalParams).

    Returns:
        Extra CLI argument string to append to the base command.
    """
    if isinstance(params, TahoeDrugEvalParams):
        return _build_tahoe_extra_args(params)
    if isinstance(params, Norman2019EvalParams):
        return _build_norman2019_extra_args(params)

    # VCCScoreParams: PARSE-specific arguments.
    extra = ""
    if params.exp_id:
        extra += f" --exp-id {params.exp_id}"
    if params.donor_id:
        extra += f" --donor-id {params.donor_id}"
    return extra


def _build_tahoe_extra_args(params: TahoeDrugEvalParams) -> str:
    """Resolve meta JSON paths and build Tahoe-specific CLI arguments.

    Args:
        params: TahoeDrugEvalParams with eval_meta_json, split, cell_line.

    Returns:
        Extra CLI argument string for Tahoe_Drug_Eval.sh.

    Raises:
        FileNotFoundError: If meta JSON or resolved paths don't exist.
        ValueError: If split/cell_line not found in meta JSON.
    """
    meta_path = Path(params.eval_meta_json)
    if not meta_path.is_absolute():
        meta_path = Path.cwd() / meta_path
    if not meta_path.exists():
        raise FileNotFoundError(f"eval_meta_json not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    split_data = meta.get("splits", {}).get(params.split)
    if split_data is None:
        raise ValueError(
            f"Split '{params.split}' not found in meta JSON. "
            f"Available: {list(meta.get('splits', {}).keys())}"
        )

    cl_data = split_data.get("cell_lines", {}).get(params.cell_line)
    if cl_data is None:
        raise ValueError(
            f"Cell line '{params.cell_line}' not found in split '{params.split}'. "
            f"Available: {list(split_data.get('cell_lines', {}).keys())}"
        )

    # Resolve paths relative to project root (paths in meta JSON are relative).
    project_root = Path.cwd()
    validation_h5ad = (project_root / cl_data["real_h5ad"]).resolve()
    real_de_csv = (project_root / cl_data["real_de_csv"]).resolve()
    baseline_agg_csv = (project_root / cl_data["baseline_agg_csv"]).resolve()

    for p, label in [
        (validation_h5ad, "validation h5ad"),
        (real_de_csv, "real_de CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    extra = (
        f" --validation {validation_h5ad}"
        f" --real-de {real_de_csv}"
        f" --profile {params.profile}"
        f" --pert-col {params.pert_col}"
        f' --control-pert "{params.control_pert}"'
        f" --celltype-col {params.celltype_col}"
    )

    if baseline_agg_csv.exists():
        extra += f" --base-agg {baseline_agg_csv}"

    print(f"  Split: {params.split}, Cell line: {params.cell_line}")
    print(f"  Validation: {validation_h5ad}")
    print(f"  Real DE: {real_de_csv}")
    print(f"  Baseline agg: {baseline_agg_csv}")

    return extra


def _build_norman2019_extra_args(params: Norman2019EvalParams) -> str:
    """Resolve Norman2019 paths and build CLI arguments.

    Unlike Tahoe, paths are specified directly in the config rather
    than looked up from a meta JSON.

    Args:
        params: Norman2019EvalParams with real_h5ad, real_de_csv,
            baseline_agg_csv paths.

    Returns:
        Extra CLI argument string for Norman2019_Double_Gene_Eval.sh.

    Raises:
        FileNotFoundError: If required paths don't exist.
    """
    project_root = Path.cwd()

    real_h5ad = Path(params.real_h5ad)
    if not real_h5ad.is_absolute():
        real_h5ad = (project_root / real_h5ad).resolve()
    real_de_csv = Path(params.real_de_csv)
    if not real_de_csv.is_absolute():
        real_de_csv = (project_root / real_de_csv).resolve()
    baseline_agg_csv = Path(params.baseline_agg_csv)
    if not baseline_agg_csv.is_absolute():
        baseline_agg_csv = (project_root / baseline_agg_csv).resolve()

    for p, label in [
        (real_h5ad, "real validation h5ad"),
        (real_de_csv, "real_de CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    extra = (
        f" --validation {real_h5ad}"
        f" --real-de {real_de_csv}"
        f" --profile {params.profile}"
        f" --pert-col {params.pert_col}"
        f' --control-pert "{params.control_pert}"'
    )

    if baseline_agg_csv.exists():
        extra += f" --base-agg {baseline_agg_csv}"

    print(f"  Eval set: {params.eval_set}")
    print(f"  Validation: {real_h5ad}")
    print(f"  Real DE: {real_de_csv}")
    print(f"  Baseline agg: {baseline_agg_csv}")

    return extra


def _parse_results(output_dir: Path) -> dict[str, Any]:
    """Parse agg_results.csv and baseline_diff.csv from eval output.

    Args:
        output_dir: Directory containing eval output files.

    Returns:
        Dict of metric results.
    """
    metrics: dict[str, Any] = {}

    agg_path = output_dir / "agg_results.csv"
    if not agg_path.exists():
        candidates = list(output_dir.glob("*_agg_results.csv"))
        if candidates:
            agg_path = candidates[0]
    if agg_path.exists():
        df = pd.read_csv(agg_path)
        if not df.empty and "statistic" in df.columns:
            mean_row = df[df["statistic"] == "mean"]
            if not mean_row.empty:
                for col in mean_row.columns:
                    if col == "statistic":
                        continue
                    val = mean_row[col].iloc[0]
                    if pd.notna(val):
                        metrics[f"agg_{col}"] = float(val)

    # Check both possible filenames for score output.
    for score_filename in ["baseline_diff_vcc_score.csv", "baseline_diff.csv"]:
        score_path = output_dir / score_filename
        if score_path.exists():
            df = pd.read_csv(score_path)
            if not df.empty and "metric" in df.columns and "from_baseline" in df.columns:
                for _, row in df.iterrows():
                    metrics[f"score_{row['metric']}"] = float(row["from_baseline"])
            break

    if not metrics:
        print(f"  WARNING: No parseable results found in {output_dir}")

    return metrics


# ---------------------------------------------------------------------------
# Registration: VCC / PARSE presets
# ---------------------------------------------------------------------------


def _make_factory(name: str):
    """Create a factory class for a named VCCScoreMetric variant."""
    class _Factory(VCCScoreMetric):
        _external = True
        def __init__(self):
            super().__init__(name=name)
    _Factory.__name__ = f"VCCScore_{name}"
    _Factory.__qualname__ = f"VCCScore_{name}"
    return _Factory


register_metric("vcc_val30")(_make_factory("vcc_val30"))
register_metric("vcc_val")(_make_factory("vcc_val"))
register_metric("vcc_test")(_make_factory("vcc_test"))
register_metric("parse_organoid")(_make_factory("parse_organoid"))
register_metric("parse_cond_val")(_make_factory("parse_cond_val"))
register_metric("parse_cond_test")(_make_factory("parse_cond_test"))


# ---------------------------------------------------------------------------
# Registration: Tahoe drug eval (3 splits × 3 cell_lines = 9 metrics)
# ---------------------------------------------------------------------------

_TAHOE_VAL_SPLITS = {
    "val_uc_pair": ["CVCL_0399", "CVCL_1693", "CVCL_1731"],
    "val_uc_dose": ["CVCL_1094", "CVCL_1119", "CVCL_1478"],
    "val_ud": ["CVCL_0131", "CVCL_1495", "CVCL_1635"],
}

for _split, _cell_lines in _TAHOE_VAL_SPLITS.items():
    for _cl in _cell_lines:
        _metric_name = f"tahoe_{_split}_{_cl}"
        register_metric(_metric_name)(_make_factory(_metric_name))


# ---------------------------------------------------------------------------
# Registration: Norman2019 double-gene perturbation eval
# ---------------------------------------------------------------------------

for _eval_set in ("norman2019_test_double", "norman2019_sanity_single",
                   "adamson_crispri_test_double"):
    register_metric(_eval_set)(_make_factory(_eval_set))
