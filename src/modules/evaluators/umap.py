"""UMAP evaluation metric: visual overlap of real vs generated cells.

Supports both CPU (scanpy) and GPU (rapids-singlecell) backends.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from . import register_metric


@register_metric("umap")
class UMAPMetric:
    """UMAP visualization comparing real and generated cells.

    Produces a 2D UMAP plot colored by batch (Real vs Generated).
    """

    name = "umap"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute UMAP embedding and save plot.

        Args:
            gen_adata: Preprocessed generated data (normalized + log1p).
            real_adata: Preprocessed real data (normalized + log1p).
            output_dir: Directory to save the UMAP PNG.
            device: ``"cpu"`` or ``"cuda"``.

        Returns:
            Empty dict (UMAP is qualitative, no scalar metric).
        """
        # Copy before mutating: gen_adata / real_adata are shared across all
        # metrics in the eval loop; writing obs["batch"] in place would leak a
        # stale column into metrics that run afterwards.
        real_adata = real_adata.copy()
        gen_adata = gen_adata.copy()
        real_adata.obs["batch"] = "Real"
        gen_adata.obs["batch"] = "Generated"
        merged = ad.concat([real_adata, gen_adata], join="inner")
        merged.obs_names_make_unique()

        if device == "cuda":
            self._compute_gpu(merged)
        else:
            self._compute_cpu(merged)

        output_path = output_dir / "umap_real_vs_gen.png"
        fig = sc.pl.umap(
            adata=merged,
            color="batch",
            size=6,
            alpha=1.0 if merged.n_obs <= 50000 else 0.3,
            title="UMAP: Real vs. Generated",
            show=False,
            return_fig=True,
        )
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")
        return {}

    def _compute_gpu(self, adata: ad.AnnData) -> None:
        """GPU-accelerated UMAP via rapids-singlecell."""
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(adata)
        rsc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata._inplace_subset_var(adata.var.highly_variable)
        rsc.pp.scale(adata)
        rsc.pp.pca(adata)
        rsc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
        rsc.tl.umap(adata)
        rsc.get.anndata_to_CPU(adata)

    def _compute_cpu(self, adata: ad.AnnData) -> None:
        """CPU UMAP via scanpy."""
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata._inplace_subset_var(adata.var.highly_variable)
        sc.pp.scale(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
        sc.tl.umap(adata)
