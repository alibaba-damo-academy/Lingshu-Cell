"""Distribution difference metrics between real and generated single-cell data.

Metrics computed:
- pearson_corr: Pearson correlation of per-gene mean expression
- spearman_corr: Spearman rank correlation of per-gene mean expression
- mean_1_WS_dist: mean per-gene 1-D Wasserstein distance
- mmd: Maximum Mean Discrepancy (RBF kernel) in PCA space
- ilisi: iLISI integration score

Reference: archive/legacy/eval/plot_metrics.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import stats

from . import register_metric


@register_metric("distribution")
class DistributionMetrics:
    """Compute distribution difference metrics."""

    name = "distribution"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute all distribution metrics and save to CSV.

        Args:
            gen_adata: Preprocessed generated data (normalized + log1p).
            real_adata: Preprocessed real data (normalized + log1p).
            output_dir: Directory to save ``distribution_metrics.csv``.
            device: ``"cpu"`` or ``"cuda"``.

        Returns:
            Dict of metric name → scalar value.
        """
        real_X = _to_dense(real_adata.X)
        gen_X = _to_dense(gen_adata.X)

        metrics: dict[str, Any] = {}

        # Per-gene mean expression correlation.
        mean_real = real_X.mean(axis=0)
        mean_gen = gen_X.mean(axis=0)

        pearson_r, _ = stats.pearsonr(mean_real, mean_gen)
        spearman_r, _ = stats.spearmanr(mean_real, mean_gen)
        metrics["pearson_corr"] = float(pearson_r)
        metrics["spearman_corr"] = float(spearman_r)

        # Mean 1-D Wasserstein distance across genes.
        print("  Computing per-gene Wasserstein distances...")
        t0 = time.time()
        if device == "cuda":
            metrics["mean_1_WS_dist"] = _wasserstein_gpu(real_X, gen_X)
        else:
            n_genes = real_X.shape[1]
            ws_dists = np.array([
                stats.wasserstein_distance(real_X[:, i], gen_X[:, i])
                for i in range(n_genes)
            ])
            metrics["mean_1_WS_dist"] = float(ws_dists.mean())
        print(f"    Wasserstein: {time.time() - t0:.1f}s")

        # MMD in PCA space.
        print("  Computing MMD in PCA space...")
        t0 = time.time()
        metrics["mmd"] = _compute_mmd(real_X, gen_X, device=device)
        print(f"    MMD total: {time.time() - t0:.1f}s")

        # iLISI.
        print("  Computing iLISI...")
        t0 = time.time()
        metrics["ilisi"] = _compute_ilisi(real_X, gen_X, device=device)
        print(f"    iLISI total: {time.time() - t0:.1f}s")

        # Save to CSV.
        df = pd.DataFrame([metrics])
        csv_path = output_dir / "distribution_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        return metrics


def _to_dense(X) -> np.ndarray:
    """Convert sparse or array-like to dense numpy array."""
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _wasserstein_gpu(real_X: np.ndarray, gen_X: np.ndarray) -> float:
    """Compute mean per-gene 1-D Wasserstein distance on GPU via cupy.

    For equal-size samples the 1-D Wasserstein distance simplifies to
    ``mean(|sort(u) - sort(v)|)``. Columns are sorted in parallel on GPU.

    Args:
        real_X: Real data matrix (n_cells, n_genes), float32.
        gen_X: Generated data matrix (n_cells, n_genes), float32.

    Returns:
        Mean Wasserstein distance averaged over all genes.
    """
    import cupy as cp

    n_genes = real_X.shape[1]
    chunk_size = 8192
    ws_chunks = []

    for col_start in range(0, n_genes, chunk_size):
        col_end = min(col_start + chunk_size, n_genes)
        r = cp.asarray(real_X[:, col_start:col_end])
        g = cp.asarray(gen_X[:, col_start:col_end])
        r_sorted = cp.sort(r, axis=0)
        g_sorted = cp.sort(g, axis=0)
        ws_chunks.append(cp.abs(r_sorted - g_sorted).mean(axis=0))
        del r, g, r_sorted, g_sorted

    ws_per_gene = cp.concatenate(ws_chunks)
    result = float(cp.mean(ws_per_gene))
    del ws_per_gene, ws_chunks
    cp.get_default_memory_pool().free_all_blocks()
    return result


# cuML's dense PCA builds an (n_features x n_features) covariance and runs a full
# cuSOLVER eigendecomposition (cusolverDnXsyevd). At genome scale (~38.6k genes)
# that covariance is ~6 GB and the eig buffer-size query fails with
# CUSOLVER_STATUS_INVALID_VALUE.
#
# This is a PERFORMANCE gate, not a correctness boundary: correctness is
# guaranteed by the try/except CPU fallback below regardless of its value. cuML
# forms the (expensive) covariance before the eig query fails, so above this many
# features we skip the doomed GPU attempt and go straight to scanpy CPU arpack —
# the SAME solver the device="cpu" path already used (so every previously-working
# path stays byte-for-byte unchanged). Set to 20000 so the ~18.1k-gene panels
# (VCC/Tahoe) still try GPU first and only fall back if it actually fails; the
# full ~38.6k-gene foundation panel is above it and goes straight to CPU.
_GPU_PCA_MAX_FEATURES = 20000


def _ensure_cpu(merged: ad.AnnData) -> None:
    """Move ``merged`` back to host memory after a failed GPU step.

    A cuML step may fail after ``anndata_to_GPU``, leaving ``X`` as a cupy array;
    bring it back so the scanpy CPU fallback does not crash. No-op if already on
    CPU or rapids_singlecell is unavailable.

    Args:
        merged: AnnData possibly holding GPU arrays (mutated in place).
    """
    try:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_CPU(merged)
    except Exception:  # noqa: BLE001 - already on CPU / rsc unavailable
        pass


def _pca_for_mmd(merged: ad.AnnData, *, device: str) -> None:
    """Write ``merged.obsm["X_pca"]`` for the MMD embedding.

    Preserves the original per-device solver exactly: GPU rapids PCA when
    ``device == "cuda"`` and the matrix is narrow enough, else scanpy CPU arpack
    (the unchanged ``device="cpu"`` call). Only genome-scale GPU PCA (which cuML
    cannot eigendecompose) or a GPU failure falls through to the CPU branch.

    Args:
        merged: Stacked real+gen AnnData; mutated to add ``obsm["X_pca"]``.
        device: ``"cuda"`` or ``"cpu"``.
    """
    # GPU path: identical to the original (no n_comps -> rapids default), only
    # gated to narrow matrices so the genome-scale covariance-eig is never built.
    if device == "cuda" and merged.n_vars <= _GPU_PCA_MAX_FEATURES:
        try:
            import rapids_singlecell as rsc

            rsc.get.anndata_to_GPU(merged)
            rsc.pp.pca(merged)
            rsc.get.anndata_to_CPU(merged)
            return
        except Exception as exc:  # noqa: BLE001 - fall back to CPU on any GPU error
            print(f"      GPU PCA failed ({type(exc).__name__}: {exc}); using CPU arpack.")
            _ensure_cpu(merged)

    # CPU path: identical to the original device="cpu" branch.
    sc.tl.pca(merged, svd_solver="arpack")


def _neighbors_for_ilisi(merged: ad.AnnData, *, device: str) -> None:
    """Build the kNN graph (``obsp``) for the iLISI score.

    Preserves the original per-device behavior exactly: GPU rapids PCA+neighbors
    when ``device == "cuda"`` and the matrix is narrow enough, else scanpy
    ``pp.neighbors`` (whose auto-PCA uses arpack) — the unchanged ``device="cpu"``
    call. Genome-scale (cuML can't PCA it) or any GPU error falls through to the
    scanpy CPU path.

    Args:
        merged: Stacked real+gen AnnData with an obs ``batch`` column; mutated to
            add the neighbors graph.
        device: ``"cuda"`` or ``"cpu"``.
    """
    # GPU path: identical to the original (rapids PCA + neighbors on device).
    if device == "cuda" and merged.n_vars <= _GPU_PCA_MAX_FEATURES:
        try:
            import rapids_singlecell as rsc

            rsc.get.anndata_to_GPU(merged)
            rsc.pp.pca(merged)
            rsc.pp.neighbors(merged, n_neighbors=10, n_pcs=20)
            rsc.get.anndata_to_CPU(merged)
            return
        except Exception as exc:  # noqa: BLE001 - fall back to CPU on any GPU error
            print(f"      GPU neighbors failed ({type(exc).__name__}: {exc}); using CPU.")
            _ensure_cpu(merged)

    # CPU path: identical to the original device="cpu" branch (neighbors auto-PCA).
    sc.pp.neighbors(merged, n_neighbors=10, n_pcs=20)


def _compute_mmd(
    real_X: np.ndarray,
    gen_X: np.ndarray,
    *,
    device: str = "cpu",
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> float:
    """Compute MMD with multi-scale RBF kernel in PCA space.

    Args:
        real_X: Real data matrix (n_cells x n_genes).
        gen_X: Generated data matrix (n_cells x n_genes).
        device: ``"cpu"`` or ``"cuda"``.
        kernel_mul: Bandwidth multiplier between scales.
        kernel_num: Number of kernel scales.

    Returns:
        MMD scalar value.
    """
    import torch

    merged = ad.AnnData(
        np.concatenate([real_X, gen_X], axis=0).astype(np.float32)
    )

    # PCA on merged data (genome-scale or GPU failure -> CPU arpack; see _pca_for_mmd).
    t0 = time.time()
    _pca_for_mmd(merged, device=device)
    print(f"      MMD PCA: {time.time() - t0:.1f}s")

    pca = merged.obsm["X_pca"]
    n_real = real_X.shape[0]
    pca = np.ascontiguousarray(pca)
    source = torch.tensor(pca[:n_real], dtype=torch.float32, device=device)
    target = torch.tensor(pca[n_real:], dtype=torch.float32, device=device)

    n_samples = source.shape[0] + target.shape[0]
    total = torch.cat([source, target], dim=0)

    # Pairwise L2 distances (batched for memory).
    t0 = time.time()
    batch_size = 200
    n = total.shape[0]
    L2_parts = []
    for i in range(0, n, batch_size):
        diff = total[i:i+batch_size].unsqueeze(1) - total.unsqueeze(0)
        L2_parts.append(diff.pow(2).sum(2).cpu())
    L2_distance = torch.cat(L2_parts, dim=0)

    # Multi-scale RBF kernel.
    bandwidth = L2_distance.sum() / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernel_val = sum(
        torch.exp(-L2_distance / (bandwidth * kernel_mul ** i))
        for i in range(kernel_num)
    )

    # MMD statistic.
    n_r = source.shape[0]
    XX = kernel_val[:n_r, :n_r].mean()
    YY = kernel_val[n_r:, n_r:].mean()
    XY = kernel_val[:n_r, n_r:].mean()
    YX = kernel_val[n_r:, :n_r].mean()
    mmd = (XX + YY - XY - YX).item()
    print(f"      MMD kernel: {time.time() - t0:.1f}s")
    return mmd


def _compute_ilisi(
    real_X: np.ndarray,
    gen_X: np.ndarray,
    *,
    device: str = "cpu",
) -> float:
    """Compute iLISI integration score via scib.

    Args:
        real_X: Real data matrix.
        gen_X: Generated data matrix.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        iLISI scalar value, or NaN if scib is unavailable.
    """
    try:
        import scib
    except ImportError:
        print("  WARNING: scib not installed, skipping iLISI.")
        return float("nan")

    merged = ad.AnnData(
        np.concatenate([real_X, gen_X], axis=0).astype(np.float32)
    )
    merged.obs["batch"] = (
        ["real"] * real_X.shape[0] + ["generated"] * gen_X.shape[0]
    )

    # Build KNN graph (genome-scale or GPU failure -> scanpy CPU; see _neighbors_for_ilisi).
    t0 = time.time()
    _neighbors_for_ilisi(merged, device=device)
    print(f"      iLISI neighbors: {time.time() - t0:.1f}s")

    # Compute iLISI score.
    t0 = time.time()
    ilisi = scib.me.ilisi_graph(merged, batch_key="batch", type_="knn")
    print(f"      iLISI graph: {time.time() - t0:.1f}s")
    return float(ilisi)
