"""Linear regression metric: per-gene mean expression scatter plot.

Computes Pearson R, Spearman rho, slope, and intercept between
real and generated per-gene mean expression vectors, and produces
a scatter plot with the fitted line.

Reference: archive/legacy/eval/plot_linregress.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy import stats

from . import register_metric


@register_metric("linregress")
class LinregressMetric:
    """Mean expression linear regression evaluation."""

    name = "linregress"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute linear regression and save scatter plot.

        Args:
            gen_adata: Preprocessed generated data (normalized + log1p).
            real_adata: Preprocessed real data (normalized + log1p).
            output_dir: Directory to save ``linregress.png``.
            device: Unused (CPU only).

        Returns:
            Dict with ``pearson_r``, ``spearman_rho``, ``slope``, ``intercept``.
        """
        real_X = _to_dense(real_adata.X)
        gen_X = _to_dense(gen_adata.X)

        mean_real = real_X.mean(axis=0)
        mean_gen = gen_X.mean(axis=0)

        pearson_r, _ = stats.pearsonr(mean_real, mean_gen)
        spearman_rho, _ = stats.spearmanr(mean_real, mean_gen)
        reg = stats.linregress(mean_real, mean_gen)

        metrics = {
            "pearson_r": float(pearson_r),
            "spearman_rho": float(spearman_rho),
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
        }

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(mean_real, mean_gen, alpha=0.3, s=4, c="steelblue")

        x_line = np.linspace(mean_real.min(), mean_real.max(), 100)
        ax.plot(x_line, reg.slope * x_line + reg.intercept, "r-", linewidth=1.5)
        ax.plot(x_line, x_line, "k--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Real (per-gene mean expression)")
        ax.set_ylabel("Generated (per-gene mean expression)")
        ax.set_title("Mean Expression: Real vs. Generated")

        text = (
            f"Pearson R = {pearson_r:.4f}\n"
            f"Spearman ρ = {spearman_rho:.4f}\n"
            f"Slope = {reg.slope:.4f}\n"
            f"Intercept = {reg.intercept:.4f}"
        )
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        output_path = output_dir / "linregress.png"
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")

        return metrics


def _to_dense(X) -> np.ndarray:
    """Convert sparse or array-like to dense numpy array."""
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)
