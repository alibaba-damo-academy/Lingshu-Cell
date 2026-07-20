"""Expressed genes statistics: mean and std of genes expressed per cell.

Reports both generated and real data statistics for direct comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import anndata as ad

from . import register_metric


@register_metric("expressed_genes")
class ExpressedGenesMetric:
    """Per-cell expressed gene count statistics."""

    name = "expressed_genes"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute mean/std of expressed genes per cell for both datasets.

        Args:
            gen_adata: Preprocessed generated data (normalized + log1p).
            real_adata: Preprocessed real data (normalized + log1p).
            output_dir: Directory to save ``expressed_genes.csv``.
            device: Unused.

        Returns:
            Dict with gen/real mean and std of expressed genes per cell.
        """
        gen_X = _to_dense(gen_adata.X)
        real_X = _to_dense(real_adata.X)

        gen_counts = (gen_X > 0).sum(axis=1)
        real_counts = (real_X > 0).sum(axis=1)

        metrics = {
            "gen_mean_expressed_genes": float(np.mean(gen_counts)),
            "gen_std_expressed_genes": float(np.std(gen_counts)),
            "real_mean_expressed_genes": float(np.mean(real_counts)),
            "real_std_expressed_genes": float(np.std(real_counts)),
        }

        df = pd.DataFrame([metrics])
        csv_path = output_dir / "expressed_genes.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        return metrics


def _to_dense(X) -> np.ndarray:
    """Convert sparse or array-like to dense numpy array."""
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)
