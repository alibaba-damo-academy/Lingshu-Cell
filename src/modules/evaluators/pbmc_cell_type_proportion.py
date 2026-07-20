"""PBMC cell-type proportion metric via marker-based Leiden annotation.

PBMC-specific: the built-in marker set and the whole comparison target the five
major PBMC lineages (T / B / NK / Monocyte / DC). It is NOT a general-tissue
metric — applying it to other tissues would require a different marker set.

Compares the cell-type composition of generated vs real PBMC cells. Following
the legacy pipeline (``archive/legacy/eval/annotate_leiden_by_markers.py``),
real and generated cells are annotated *jointly* and with the *same* method, so
the comparison reflects composition differences rather than annotation-method
bias:

1. concat gen + real (a ``source`` obs column marks origin);
2. HVG → scale → PCA → neighbors → Leiden clustering on the joint space;
3. score each Leiden cluster by mean marker-gene expression and assign it to the
   cell type with the highest score (``idxmax``);
4. map clusters → ``cell_type_pred`` and compute per-source proportions;
5. report L1 distance and Pearson/Spearman correlation between the real and
   generated proportion vectors.

The real data's own ground-truth labels (``obs.cell_type``, fine-grained) are
deliberately NOT used as the reference, because they are not on the same
(coarse, marker-defined) label set as the generated annotation; using one joint
annotation keeps both sides comparable (mirrors the legacy choice).

Registered as ``pbmc_cell_type_proportion`` (standard metric: receives
preprocessed ``gen_adata`` / ``real_adata`` from the eval workflow). Runs on CPU
via scanpy; scale fits the ~20k-cell joint matrix comfortably.

Reference: ``archive/legacy/eval/annotate_leiden_by_markers.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from . import register_metric

# 内置 marker 集合（来自 legacy annotate_leiden_by_markers.py 的 pbmc_basic）。
# 5 大 PBMC lineage，对应 v1 论文 Fig2a/2b 的 major cell types。
BUILTIN_MARKERS: Dict[str, Dict[str, List[str]]] = {
    "pbmc_basic": {
        "T": ["CD3D", "CD3E", "TRAC", "IL7R", "LTB"],
        "B": ["MS4A1", "CD79A", "CD74", "HLA-DRA"],
        "NK": ["NKG7", "KLRD1", "GNLY", "FCGR3A"],
        "Monocyte": ["LYZ", "S100A8", "S100A9", "LGALS3", "FCN1"],
        "DC": ["FCER1A", "CST3", "CLEC10A", "FLT3"],
    },
}

# Leiden / 预处理超参（对齐 legacy 默认值）。
HVG_KW = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
SCALE_MAX = 10.0
N_NEIGHBORS = 10
LEIDEN_RESOLUTION = 1.0


def _resolve_markers(markers: Any) -> Dict[str, List[str]]:
    """Resolve the ``markers`` kwarg into a ``{cell_type: [genes]}`` dict.

    Args:
        markers: Either a built-in set name (str, e.g. ``"pbmc_basic"``) or an
            explicit ``{cell_type: [gene, ...]}`` mapping.

    Returns:
        The marker mapping.

    Raises:
        KeyError: If a built-in name is given but not defined.
        TypeError: If ``markers`` is neither a str nor a dict.
    """
    if isinstance(markers, str):
        if markers not in BUILTIN_MARKERS:
            raise KeyError(
                f"Unknown built-in marker set '{markers}'. "
                f"Available: {list(BUILTIN_MARKERS)}"
            )
        return BUILTIN_MARKERS[markers]
    if isinstance(markers, dict):
        return {k: list(v) for k, v in markers.items()}
    raise TypeError(f"markers must be a str or dict, got {type(markers)}")


def _marker_score_matrix(
    adata: ad.AnnData, markers: Dict[str, List[str]]
) -> pd.DataFrame:
    """Per-cell mean marker-gene expression for each cell type.

    Must be called on the (log1p, full-gene) expression BEFORE HVG subsetting,
    since marker genes are not guaranteed to survive HVG selection.

    Args:
        adata: Joint AnnData with log1p expression in ``.X`` and gene symbols as
            ``var_names``.
        markers: ``{cell_type: [gene, ...]}`` mapping.

    Returns:
        DataFrame indexed by ``adata.obs_names`` with one column per cell type
        holding the per-cell mean expression of that type's present markers.
        Cell types with no present markers get a zero column.
    """
    scores: Dict[str, np.ndarray] = {}
    for cell_type, genes in markers.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            print(f"  [pbmc_cell_type_proportion] {cell_type}: markers absent from "
                  f"aligned gene set, skipped: {missing}")
        if not present:
            print(f"  [pbmc_cell_type_proportion] WARNING: {cell_type} has no usable "
                  f"markers; its score is all-zero.")
            scores[cell_type] = np.zeros(adata.n_obs, dtype=np.float64)
            continue
        # 取该 cell type 在场 marker 基因的子矩阵并对基因维取均值（log1p 尺度）。
        sub = adata[:, present].X
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        scores[cell_type] = np.asarray(sub, dtype=np.float64).mean(axis=1).ravel()
    return pd.DataFrame(scores, index=adata.obs_names.to_numpy())


def _joint_leiden(adata: ad.AnnData) -> None:
    """HVG → scale → PCA → neighbors → Leiden on ``adata`` in place.

    Mirrors the legacy ``_preprocess_and_leiden`` defaults. Writes
    ``obs["leiden"]``. Assumes ``adata.X`` is already normalized + log1p.

    Args:
        adata: Joint (gen+real) AnnData; mutated (HVG-subset, scaled, PCA'd).
    """
    sc.pp.highly_variable_genes(adata, **HVG_KW)
    adata._inplace_subset_var(adata.var["highly_variable"])
    sc.pp.scale(adata, max_value=SCALE_MAX)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS)
    # flavor="igraph" 是 legacy 用法；旧版 scanpy 不支持该参数时退回默认 leidenalg。
    try:
        sc.tl.leiden(
            adata,
            resolution=LEIDEN_RESOLUTION,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except TypeError:
        sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION)


@register_metric("pbmc_cell_type_proportion")
class PBMCCellTypeProportionMetric:
    """PBMC marker-based joint Leiden annotation + real-vs-gen proportion."""

    name = "pbmc_cell_type_proportion"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Annotate gen+real PBMC jointly and compare cell-type proportions.

        Args:
            gen_adata: Preprocessed generated data (normalized + log1p, gene
                symbols as var_names).
            real_adata: Preprocessed real data (same preprocessing).
            output_dir: Directory to save ``proportions.csv`` and the bar chart.
            device: Unused (CPU scanpy pipeline); kept for protocol compatibility.
            **kwargs: Optional overrides:
                markers (str | dict): built-in set name or explicit mapping
                    (default ``"pbmc_basic"``).

        Returns:
            Dict with ``proportion_l1``, ``proportion_pearson``,
            ``proportion_spearman``, ``n_celltypes``, plus per-cell-type
            ``gen_<ct>`` / ``real_<ct>`` proportions.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markers = _resolve_markers(kwargs.get("markers", "pbmc_basic"))
        cell_types = list(markers.keys())

        # --- 合并 gen + real，标注来源 ---
        real_c = real_adata.copy()
        gen_c = gen_adata.copy()
        real_c.obs["source"] = "real"
        gen_c.obs["source"] = "gen"
        joint = ad.concat([real_c, gen_c], join="outer")
        joint.obs_names_make_unique()
        print(f"  [pbmc_cell_type_proportion] joint = {joint.n_obs} cells "
              f"({real_c.n_obs} real + {gen_c.n_obs} gen) x {joint.n_vars} genes")

        # --- marker 打分必须在 HVG subset 前（全基因 log1p 尺度）---
        score_df = _marker_score_matrix(joint, markers)

        # --- 联合聚类（会原地修改 joint.X：HVG subset + scale）---
        _joint_leiden(joint)

        # --- 按 marker 平均表达把每个 Leiden cluster 指派到某个 cell type ---
        score_df["leiden"] = joint.obs["leiden"].to_numpy()
        cluster_mean = score_df.groupby("leiden", observed=True)[cell_types].mean()
        cluster_to_celltype = cluster_mean.idxmax(axis=1).to_dict()
        joint.obs["cell_type_pred"] = (
            joint.obs["leiden"].map(cluster_to_celltype).astype(str)
        )

        # --- 逐来源统计 cell type 比例（行归一化）---
        cross = pd.crosstab(joint.obs["source"], joint.obs["cell_type_pred"])
        cross = cross.reindex(columns=cell_types, fill_value=0)
        prop = cross.div(cross.sum(axis=1), axis=0)

        p_real = prop.loc["real", cell_types].to_numpy(dtype=np.float64)
        p_gen = prop.loc["gen", cell_types].to_numpy(dtype=np.float64)

        # --- 标量指标：L1 距离 + 比例向量的相关 ---
        l1 = float(np.abs(p_gen - p_real).sum())
        pearson = float(stats.pearsonr(p_gen, p_real)[0]) if len(cell_types) > 1 else float("nan")
        spearman = float(stats.spearmanr(p_gen, p_real)[0]) if len(cell_types) > 1 else float("nan")

        # --- 保存 proportions.csv + 条形图 ---
        prop_out = prop.loc[["real", "gen"], cell_types]
        prop_out.to_csv(output_dir / "proportions.csv")
        cluster_mean.to_csv(output_dir / "leiden_marker_scores.csv")
        self._save_barplot(p_real, p_gen, cell_types, output_dir)

        # --- 汇总返回 ---
        results: dict[str, Any] = {
            "proportion_l1": l1,
            "proportion_pearson": pearson,
            "proportion_spearman": spearman,
            "n_celltypes": len(cell_types),
            "n_leiden_clusters": int(joint.obs["leiden"].nunique()),
        }
        for i, ct in enumerate(cell_types):
            results[f"real_{ct}"] = float(p_real[i])
            results[f"gen_{ct}"] = float(p_gen[i])
        print(f"  [pbmc_cell_type_proportion] L1={l1:.4f} pearson={pearson:.4f} "
              f"spearman={spearman:.4f}")
        return results

    def _save_barplot(
        self,
        p_real: np.ndarray,
        p_gen: np.ndarray,
        cell_types: List[str],
        output_dir: Path,
    ) -> None:
        """Save a grouped bar chart of real vs gen cell-type proportions.

        Args:
            p_real: Real proportions, aligned to ``cell_types``.
            p_gen: Generated proportions, aligned to ``cell_types``.
            cell_types: Cell-type labels (x-axis order).
            output_dir: Directory for ``proportion_barplot.png``.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(cell_types))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(4, len(cell_types) * 1.1), 3.2))
        ax.bar(x - width / 2, p_real, width, label="real", color="#4878b0")
        ax.bar(x + width / 2, p_gen, width, label="gen", color="#ee854a")
        ax.set_xticks(x)
        ax.set_xticklabels(cell_types, rotation=30, ha="right")
        ax.set_ylabel("proportion")
        ax.set_title("PBMC cell-type proportion: real vs generated")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(output_dir / "proportion_barplot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
