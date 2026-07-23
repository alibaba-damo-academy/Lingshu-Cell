"""Eval workflow: unified evaluation pipeline for generated single-cell data.

Loads generated and real data, preprocesses once, then runs selected
metric modules. Results are saved under ``{sample_dir}/eval/``.

Usage::

    python -m workflows.eval.main \\
        sample_dir=/path/to/sample_dir \\
        real_data=/path/to/real.h5ad \\
        metrics="[umap,distribution,linregress]"
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modules.evaluators import get_metric, METRIC_REGISTRY
from workflows.eval.config_schema import EvalConfig

# Import metric modules to trigger registration.
import src.modules.evaluators.umap  # noqa: F401
import src.modules.evaluators.distribution  # noqa: F401
import src.modules.evaluators.linregress  # noqa: F401
import src.modules.evaluators.expressed_genes  # noqa: F401
import src.modules.evaluators.external.vcc_score  # noqa: F401


@hydra.main(version_base=None, config_path="../../configs", config_name="eval/default")
def main(cfg: EvalConfig) -> int:
    """Hydra entry point for the eval workflow."""
    eval_workflow(cfg)
    return 0


def eval_workflow(cfg: EvalConfig) -> None:
    """Run the evaluation pipeline.

    Steps:
    1. Load generated data (merged.h5ad from sample_dir).
    2. Load real data (backed mode for large files).
    3. Preprocess: Ensembl conversion, gene alignment, sampling, normalize.
    4. Run each requested metric module.
    5. Save summary.
    """
    sample_dir = Path(cfg.sample_dir)
    device = _resolve_device(cfg.device)
    np.random.seed(cfg.seed)

    # Check which metrics need standard data loading vs external-only.
    standard_metrics = [m for m in cfg.metrics if not getattr(get_metric(m), "_external", False)]

    print(f"=== Eval Workflow ===")
    print(f"Sample dir: {sample_dir}")
    print(f"Device:     {device}")
    print(f"Metrics:    {cfg.metrics}")
    print()

    gen_adata = None
    real_adata = None

    # Load and preprocess data only if standard metrics are requested.
    if standard_metrics:
        gen_path = sample_dir / "merged.h5ad"
        if not gen_path.exists():
            raise FileNotFoundError(f"Generated data not found: {gen_path}")

        real_path = Path(cfg.real_data)
        if not real_path.exists():
            raise FileNotFoundError(f"Real data not found: {real_path}")

        print(f"Real data:  {real_path}")

        t0 = time.time()
        print("Loading generated data...")
        gen_adata = _load_h5ad_lightweight(gen_path)
        print(f"  {gen_adata.n_obs} cells x {gen_adata.n_vars} genes ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print("Loading real data...")
        real_adata = _load_h5ad_lightweight(real_path)
        print(f"  {real_adata.n_obs} cells x {real_adata.n_vars} genes ({time.time()-t0:.1f}s)")

        # --- Sampling (before alignment, to exit backed mode) ---
        gen_adata = _sample_cells(gen_adata, cfg.num_samples)
        real_adata = _sample_cells(real_adata, cfg.num_samples)

        # --- Ensembl → gene name conversion ---
        if cfg.convert_ensembl_to_name and cfg.gene_id_name_file:
            real_adata = _convert_ensembl_to_name(real_adata, cfg.gene_id_name_file)

        # --- Gene alignment ---
        gen_adata, real_adata = _align_genes(gen_adata, real_adata)

        # --- Normalize ---
        sc.pp.normalize_total(real_adata)
        sc.pp.log1p(real_adata)
        sc.pp.normalize_total(gen_adata)
        sc.pp.log1p(gen_adata)

        print(f"\nAfter preprocessing: gen={gen_adata.shape}, real={real_adata.shape}")

    # --- Run metrics ---
    eval_dir = sample_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    for metric_name in cfg.metrics:
        print(f"\n--- [{metric_name}] ---")
        metric_cls = get_metric(metric_name)
        metric = metric_cls()

        output_dir = eval_dir / metric_name
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        is_external = getattr(metric_cls, "_external", False)
        if is_external:
            ext_entry = dict(cfg.external.get(metric_name, {}))
            predict_filename = ext_entry.pop("predict_filename", "merged.h5ad")
            params_cfg = ext_entry.pop("params", {})
            params = instantiate(params_cfg)
            ext_h5ad = sample_dir / predict_filename
            results = metric.compute(
                gen_h5ad_path=ext_h5ad,
                output_dir=output_dir,
                params=params,
            )
        else:
            results = metric.compute(
                gen_adata=gen_adata,
                real_adata=real_adata,
                output_dir=output_dir,
                device=device,
            )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        if results:
            all_results[metric_name] = results
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")

    # --- Save summary ---
    summary_path = eval_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # --- 通知结果仓库:该 summary.json 所属数据集已变脏(供 reader 读时自愈刷新)---
    # best-effort:仅当该路径落在某个已注册仓库根下才生效;否则(VCC/PARSE 等)静默 no-op。
    # 失败不影响 eval 主流程。
    try:
        from src.results_warehouse import mark_dirty_for_path

        mark_dirty_for_path(summary_path)
    except Exception as exc:  # noqa: BLE001 - 仓库通知是旁路,不应让 eval 失败
        print(f"[results_warehouse] mark_dirty skipped: {exc}")


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def _load_h5ad_lightweight(path: Path) -> ad.AnnData:
    """Load h5ad in backed mode, then sample into memory.

    For large files, reads only obs/var metadata from disk; the full
    AnnData is loaded into memory only after sampling.
    """
    import h5py
    try:
        from anndata._io.specs import read_elem
    except ImportError:
        from anndata.experimental import read_elem

    with h5py.File(str(path), "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])

    shape = (len(obs), len(var))
    adata = ad.AnnData(obs=obs, var=var, shape=shape, filename=str(path), filemode="r")
    return adata


def _convert_ensembl_to_name(adata: ad.AnnData, gene_id_name_file: str) -> ad.AnnData:
    """Convert var_names from Ensembl IDs to gene names."""
    df = pd.read_csv(gene_id_name_file, sep="\t", header=None, names=["ensembl_id", "gene_name"])
    id_to_name = dict(zip(df["ensembl_id"], df["gene_name"]))

    original_ids = adata.var_names.tolist()
    new_names = [id_to_name.get(gid, gid) for gid in original_ids]
    mapped = sum(1 for old, new in zip(original_ids, new_names) if old != new)
    print(f"  Ensembl → gene name: {mapped}/{len(original_ids)} mapped")

    adata.var_names = new_names
    return adata


def _align_genes(gen_adata: ad.AnnData, real_adata: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    """Align two AnnData objects to their shared gene set."""
    common = sorted(set(gen_adata.var_names) & set(real_adata.var_names))
    print(f"  Gene alignment: gen={gen_adata.n_vars}, real={real_adata.n_vars}, common={len(common)}")
    if len(common) == 0:
        raise ValueError("No common genes between generated and real data.")
    return gen_adata[:, common].copy(), real_adata[:, common].copy()


def _sample_cells(adata: ad.AnnData, num_samples: int) -> ad.AnnData:
    """Sample cells from AnnData. Handles backed mode via to_memory()."""
    is_backed = adata.isbacked
    if adata.n_obs <= num_samples:
        return adata.to_memory() if is_backed else adata.copy()
    indices = np.random.choice(adata.n_obs, size=num_samples, replace=False)
    subset = adata[indices]
    return subset.to_memory() if is_backed else subset.copy()


def _resolve_device(device_str: str) -> str:
    """Resolve device string."""
    if device_str == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


if __name__ == "__main__":
    raise SystemExit(main())
