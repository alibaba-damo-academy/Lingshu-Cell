"""Batch-integration metric for the foundation eval (Track A, CxG).

Designed for the redesigned batch-effect eval (see
``.docs/pretrain/batch_effect_eval_design_v1.html``). Unlike the older
``batch_effect`` metric (single fixed batch_id, raw-space comparison), this one
assumes the generated cells span the SAME multiple real batches as the real
reference (the condition CSV mirrors real ``(batch_id, cell_description)``
pairs), and it measures generation fidelity + integration in the
Harmony-corrected joint space — the scIB-style "query vs reference" setup.

SCOPE — what this DOES and does NOT test:

- DOES: generation fidelity + integration. Does the generated set blend into the
  real manifold after batch correction (``a2_ilisi_datasource`` high) while
  preserving cell types (``a2_clisi_celltype`` high)? Generated cells carry the
  real cells' OWN batch_ids, so batch is *matched* to real, not held fixed.
- Does NOT: the model's batch-UNIFICATION capability (whether ``batch_id`` is a
  removable technical knob — i.e. fixing biology and varying only batch produces
  cells that collapse together). That test needs biology ⊥ batch, which holds on
  the Tahoe plate substrate (every cell line on every plate) but NOT on CxG
  (cell_description bakes in donor/disease/assay, so it correlates with batch_id).
  The batch-unification test therefore lives on the separate Tahoe track.

Two analyses on one tissue / study subset:

- **A1 (real only)** — scIB baseline/ceiling on the real cells alone:
  ``iLISI(batch_id)`` raw vs Harmony (low→high = batch effect present and
  removable) and ``cLISI(cell_type)`` (should stay high = biology conserved).

Score convention (scib ``me`` API, ``scale=True``): BOTH iLISI and cLISI are
rescaled to [0, 1] where **higher = better** — iLISI high = batches well mixed;
cLISI high = cell types well separated (i.e. neighborhoods are pure).
- **A2 (real + generated)** — co-embed real+gen, Harmony on ``batch_id``, then
  measure in the corrected space: ``iLISI(data_source)`` (gen vs real mixing =
  fidelity), ``cLISI(cell_type)`` (biology conserved), ``iLISI(batch_id)``. Raw
  (uncorrected) versions are reported alongside for reference. A UMAP overlay of
  both the corrected and raw embeddings is saved, colored by
  ``data_source`` / ``batch_id`` / ``cell_type``.

Heavy genome-scale steps (HVG/scale/PCA/Harmony/UMAP) run on GPU via
``rapids_singlecell`` when ``device="cuda"`` (with a scanpy CPU fallback);
iLISI/cLISI use scib on the resulting kNN graph. HVG selection precedes PCA, so
the covariance is small and the genome-scale cuSOLVER issue (see
``evaluators/distribution.py``) does not arise here. Run in the ``rapids_sc``
env (rapids_singlecell + scib). Registered as ``batch_integration``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from . import register_metric

CELL_TYPE_KEY = "cell_type_coarse"
BATCH_KEY = "batch_id"
HVG_KW = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
N_PCS = 20
N_NEIGHBORS = 15


def _ensure_cpu(adata: ad.AnnData) -> None:
    """Move ``adata`` back to host memory after a failed GPU step.

    A cuML step may fail after ``anndata_to_GPU``, leaving ``X`` as a cupy
    array; bring it back so the scanpy CPU fallback does not crash. No-op if
    already on CPU or rapids_singlecell is unavailable.

    Args:
        adata: AnnData possibly holding GPU arrays (mutated in place).
    """
    try:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_CPU(adata)
    except Exception:  # noqa: BLE001 - already on CPU / rsc unavailable
        pass


def _embed(adata: ad.AnnData, *, device: str, batch_key: str, with_harmony: bool) -> bool:
    """HVG → scale → PCA (+ optional Harmony) on ``adata`` in place.

    GPU via rapids_singlecell when ``device == "cuda"``, else scanpy CPU; any GPU
    error falls back to CPU. Writes ``obsm["X_pca"]`` and, when *with_harmony*,
    ``obsm["X_pca_harmony"]``.

    Args:
        adata: Normalized+log1p AnnData (mutated: HVG-subset, scaled).
        device: ``"cuda"`` (rapids_singlecell) or else scanpy CPU.
        batch_key: obs column for Harmony integration.
        with_harmony: Whether to also compute the Harmony-corrected rep.

    Returns:
        ``True`` if the Harmony rep was produced, else ``False``.
    """
    harmony_done = False
    if device == "cuda":
        try:
            import rapids_singlecell as rsc

            rsc.get.anndata_to_GPU(adata)
            rsc.pp.highly_variable_genes(adata, **HVG_KW)
            adata._inplace_subset_var(adata.var.highly_variable)
            rsc.pp.scale(adata)
            rsc.pp.pca(adata, n_comps=N_PCS)
            if with_harmony:
                rsc.pp.harmony_integrate(adata, key=batch_key, adjusted_basis="X_pca_harmony")
                harmony_done = "X_pca_harmony" in adata.obsm
            rsc.get.anndata_to_CPU(adata)
            return harmony_done
        except Exception as exc:  # noqa: BLE001 - fall back to CPU on any GPU error
            print(f"  GPU embed failed ({exc}); falling back to scanpy CPU.")
            _ensure_cpu(adata)

    # CPU path.
    sc.pp.highly_variable_genes(adata, **HVG_KW)
    adata._inplace_subset_var(adata.var.highly_variable)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=N_PCS)
    if with_harmony:
        try:
            import scanpy.external as sce

            sce.pp.harmony_integrate(adata, key=batch_key, adjusted_basis="X_pca_harmony")
            harmony_done = "X_pca_harmony" in adata.obsm
        except Exception as exc:  # noqa: BLE001 - harmonypy may be absent on CPU
            print(f"  CPU Harmony unavailable ({exc}); skipping corrected rep.")
    return harmony_done


def _lisi(adata: ad.AnnData, *, rep: str, key: str, kind: str) -> float:
    """Build a kNN graph on ``rep`` and return the scib iLISI/cLISI score.

    Args:
        adata: AnnData with ``obsm[rep]`` and the obs ``key`` column.
        rep: ``obsm`` key to build neighbors on (e.g. ``"X_pca"``).
        key: obs column to score over.
        kind: ``"ilisi"`` (integration) or ``"clisi"`` (cell-type separation).
            Under scib's scaled ``me`` API both are in [0, 1] with higher=better
            (iLISI high = batches mixed; cLISI high = cell types separated).

    Returns:
        The LISI scalar, or ``float("nan")`` if scib is unavailable or the key
        has <2 categories (LISI undefined).
    """
    try:
        import scib
    except ImportError:
        print(f"  WARNING: scib not installed, skipping {kind}({key}).")
        return float("nan")
    if key not in adata.obs or adata.obs[key].nunique() < 2:
        return float("nan")
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=N_NEIGHBORS)
    if kind == "ilisi":
        return float(scib.me.ilisi_graph(adata, batch_key=key, type_="knn"))
    return float(scib.me.clisi_graph(adata, label_key=key, type_="knn"))


@register_metric("batch_integration")
class BatchIntegrationMetrics:
    """scIB-style batch-integration metrics (A1 real-only + A2 real+gen)."""

    name = "batch_integration"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute Track-A batch-integration metrics + UMAP overlays.

        Args:
            gen_adata: Generated cells; obs must carry ``batch_id`` (and
                ``cell_type_coarse`` when available) via the infer obs
                passthrough.
            real_adata: Real reference cells with ``batch_id`` /
                ``cell_type_coarse``.
            output_dir: Directory for the UMAP overlay PNGs.
            device: ``"cuda"`` (rapids_singlecell) or ``"cpu"``.

        Returns:
            Dict of metric name → scalar (NaN where a dep/column is missing).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Validate obs contract ---
        for label, a in (("real", real_adata), ("gen", gen_adata)):
            if BATCH_KEY not in a.obs:
                raise ValueError(f"batch_integration: {label}_adata missing obs '{BATCH_KEY}'.")
        has_celltype = (
            CELL_TYPE_KEY in real_adata.obs.columns and CELL_TYPE_KEY in gen_adata.obs.columns
        )
        if not has_celltype:
            print(f"  WARNING: '{CELL_TYPE_KEY}' missing on gen/real; cLISI skipped.")

        n_batches = int(real_adata.obs[BATCH_KEY].nunique())
        metrics: dict[str, Any] = {
            "n_real": int(real_adata.n_obs),
            "n_gen": int(gen_adata.n_obs),
            "n_batches": n_batches,
        }
        if has_celltype:
            metrics["n_cell_types"] = int(real_adata.obs[CELL_TYPE_KEY].nunique())

        # ---------- A1: real only (baseline + Harmony ceiling) ----------
        print("  [A1] embedding real (raw + Harmony)...")
        t0 = time.time()
        A = real_adata.copy()
        harmony_ok = _embed(A, device=device, batch_key=BATCH_KEY, with_harmony=(n_batches >= 2))
        metrics["a1_ilisi_batch__raw"] = _lisi(A, rep="X_pca", key=BATCH_KEY, kind="ilisi")
        if has_celltype:
            metrics["a1_clisi_celltype__raw"] = _lisi(A, rep="X_pca", key=CELL_TYPE_KEY, kind="clisi")
        if harmony_ok:
            metrics["a1_ilisi_batch__harmony"] = _lisi(A, rep="X_pca_harmony", key=BATCH_KEY, kind="ilisi")
            if has_celltype:
                metrics["a1_clisi_celltype__harmony"] = _lisi(
                    A, rep="X_pca_harmony", key=CELL_TYPE_KEY, kind="clisi"
                )
        else:
            metrics["a1_ilisi_batch__harmony"] = float("nan")
        print(f"    A1 done ({time.time()-t0:.1f}s)")

        # ---------- A2: real + generated (integration / fidelity) ----------
        print("  [A2] embedding real + generated (Harmony on batch_id)...")
        t0 = time.time()
        real_c = real_adata.copy()
        gen_c = gen_adata.copy()
        real_c.obs["data_source"] = "real"
        gen_c.obs["data_source"] = "gen"
        AC = ad.concat([real_c, gen_c], join="inner")
        AC.obs_names_make_unique()
        ac_harmony_ok = _embed(AC, device=device, batch_key=BATCH_KEY, with_harmony=(n_batches >= 2))

        # Raw-space metrics (reference): mixing/biology before batch correction.
        metrics["a2_ilisi_datasource__raw"] = _lisi(AC, rep="X_pca", key="data_source", kind="ilisi")
        metrics["a2_ilisi_batch__raw"] = _lisi(AC, rep="X_pca", key=BATCH_KEY, kind="ilisi")
        if has_celltype:
            metrics["a2_clisi_celltype__raw"] = _lisi(AC, rep="X_pca", key=CELL_TYPE_KEY, kind="clisi")

        # Corrected-space metrics (primary): the scIB query-vs-reference read —
        # batch removed, so data_source mixing reflects biological fidelity.
        if ac_harmony_ok:
            metrics["a2_ilisi_datasource__corrected"] = _lisi(
                AC, rep="X_pca_harmony", key="data_source", kind="ilisi"
            )
            metrics["a2_ilisi_batch__corrected"] = _lisi(
                AC, rep="X_pca_harmony", key=BATCH_KEY, kind="ilisi"
            )
            if has_celltype:
                metrics["a2_clisi_celltype__corrected"] = _lisi(
                    AC, rep="X_pca_harmony", key=CELL_TYPE_KEY, kind="clisi"
                )
        else:
            metrics["a2_ilisi_datasource__corrected"] = float("nan")
        print(f"    A2 done ({time.time()-t0:.1f}s)")

        # ---------- UMAP overlays (corrected + raw) ----------
        try:
            colorings = ["data_source", BATCH_KEY] + ([CELL_TYPE_KEY] if has_celltype else [])
            if ac_harmony_ok:
                self._save_umap(AC, output_dir, device=device, rep="X_pca_harmony",
                                tag="corrected", colorings=colorings)
            self._save_umap(AC, output_dir, device=device, rep="X_pca",
                            tag="raw", colorings=colorings)
        except Exception as exc:  # noqa: BLE001 - plotting must not fail the metric
            print(f"  WARNING: UMAP overlay failed: {exc}")

        return metrics

    def _save_umap(
        self, AC: ad.AnnData, output_dir: Path, *, device: str, rep: str,
        tag: str, colorings: list,
    ) -> None:
        """Compute one joint UMAP on ``rep`` and save overlays by each coloring.

        Args:
            AC: Joint real+gen AnnData (already embedded; has ``obsm[rep]``).
            output_dir: Directory for the PNGs.
            device: ``"cuda"`` or ``"cpu"``.
            rep: ``obsm`` key to build neighbors+UMAP on (``X_pca_harmony`` or
                ``X_pca``).
            tag: Filename tag (``"corrected"`` / ``"raw"``).
            colorings: obs columns to color the shared embedding by.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # neighbors + UMAP on the chosen rep (GPU when available, else scanpy).
        if device == "cuda":
            try:
                import rapids_singlecell as rsc

                rsc.get.anndata_to_GPU(AC)
                rsc.pp.neighbors(AC, use_rep=rep, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
                rsc.tl.umap(AC)
                rsc.get.anndata_to_CPU(AC)
            except Exception as exc:  # noqa: BLE001
                print(f"  GPU UMAP failed ({exc}); using scanpy CPU.")
                _ensure_cpu(AC)
                sc.pp.neighbors(AC, use_rep=rep, n_neighbors=N_NEIGHBORS)
                sc.tl.umap(AC)
        else:
            sc.pp.neighbors(AC, use_rep=rep, n_neighbors=N_NEIGHBORS)
            sc.tl.umap(AC)

        # One shared embedding, one PNG per coloring.
        for color in colorings:
            legend = "right margin" if AC.obs[color].nunique() <= 25 else None
            fig = sc.pl.umap(
                AC, color=color, show=False, return_fig=True,
                size=6, alpha=0.6 if AC.n_obs > 20000 else 1.0,
                title=f"A∪C ({tag}) — {color}", legend_loc=legend,
            )
            fig.savefig(str(output_dir / f"umap_{tag}_{color}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved UMAP overlays [{tag}] ({', '.join(colorings)}) -> {output_dir}")
