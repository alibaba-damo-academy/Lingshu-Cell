"""Batch-effect unification metric for the foundation R3 evaluation (Eval2).

Run PER TISSUE (one tissue per eval run). Within a tissue, the same cell types
recur across many real technical batches, so "batch effect" is well-defined and
removable. The test: generate biologically diverse cells (real cell_description
+ perturbation_description) all under ONE fixed ``batch_id``; if ``batch_id`` is a
pure technical knob, the generated cells should look like the real data AFTER
batch correction — batches mixed, cell types preserved.

Three views, compared via iLISI / cLISI on a kNN graph:

- **A** = real (raw): expect LOW iLISI(batch_id) — batch effect present (baseline).
- **B** = real + Harmony: expect HIGH iLISI(batch_id) — de-batched ceiling.
- **A∪C** = real + generated(fixed batch): does adding C raise iLISI toward B
  while keeping cLISI(cell_type) low (biology preserved)? Plus
  iLISI(data_source) = are generated cells indistinguishable from real.

Heavy genome-scale steps (HVG/scale/PCA/Harmony/UMAP) run on GPU via
``rapids_singlecell`` when ``device="cuda"`` (mirrors ``evaluators/umap.py``),
with a scanpy CPU fallback. iLISI/cLISI use scib on the resulting kNN graph.
UMAP overlay of the A∪C joint embedding is saved colored by data_source /
batch_id / cell_type. Run in the ``rapids_sc`` env (has rapids_singlecell + scib).

Registered as ``batch_effect`` (standard metric: gets preprocessed
``gen_adata`` / ``real_adata`` from the eval workflow).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import numpy as np
import scanpy as sc

from . import register_metric

CELL_TYPE_KEY = "cell_type_coarse"
HVG_KW = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
N_PCS = 20
N_NEIGHBORS = 15


def _embed(adata: ad.AnnData, *, device: str, batch_key: str, with_harmony: bool) -> bool:
    """HVG → scale → PCA (+Harmony) on ``adata`` in place; GPU if available.

    Mirrors the repo's GPU UMAP idiom (``evaluators/umap.py``). Writes
    ``obsm["X_pca"]`` and, when *with_harmony*, ``obsm["X_pca_harmony"]``.

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
                rsc.pp.harmony_integrate(
                    adata, key=batch_key, adjusted_basis="X_pca_harmony"
                )
                harmony_done = "X_pca_harmony" in adata.obsm
            rsc.get.anndata_to_CPU(adata)
            return harmony_done
        except Exception as exc:  # noqa: BLE001 - fall back to CPU on any GPU error
            print(f"  GPU embed failed ({exc}); falling back to scanpy CPU.")
            # A GPU step may have failed AFTER anndata_to_GPU, leaving X as cupy;
            # bring it back to CPU so the scanpy fallback below does not crash.
            try:
                import rapids_singlecell as rsc

                rsc.get.anndata_to_CPU(adata)
            except Exception:  # noqa: BLE001 - already on CPU / rsc unavailable
                pass

    sc.pp.highly_variable_genes(adata, **HVG_KW)
    adata._inplace_subset_var(adata.var.highly_variable)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=N_PCS)
    if with_harmony:
        try:
            import scanpy.external as sce

            sce.pp.harmony_integrate(
                adata, key=batch_key, adjusted_basis="X_pca_harmony"
            )
            harmony_done = "X_pca_harmony" in adata.obsm
        except Exception as exc:  # noqa: BLE001 - harmonypy may be absent on CPU
            print(f"  CPU Harmony unavailable ({exc}); skipping B group.")
    return harmony_done


def _lisi(adata: ad.AnnData, *, rep: str, key: str, kind: str) -> float:
    """Build a kNN graph on ``rep`` and return the scib iLISI/cLISI score.

    Args:
        adata: AnnData with ``obsm[rep]`` and the obs ``key`` column.
        rep: ``obsm`` key to build neighbors on (e.g. ``"X_pca"``).
        key: obs column to score over.
        kind: ``"ilisi"`` (integration; higher=better) or ``"clisi"``
            (cell-type separation; lower=better).

    Returns:
        The LISI scalar, or ``float("nan")`` if scib is unavailable.
    """
    try:
        import scib
    except ImportError:
        print(f"  WARNING: scib not installed, skipping {kind}({key}).")
        return float("nan")
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=N_NEIGHBORS)
    if kind == "ilisi":
        return float(scib.me.ilisi_graph(adata, batch_key=key, type_="knn"))
    return float(scib.me.clisi_graph(adata, label_key=key, type_="knn"))


@register_metric("batch_effect")
class BatchEffectMetrics:
    """Per-tissue batch-effect unification metrics (iLISI/cLISI A/B/A∪C + UMAP)."""

    name = "batch_effect"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute batch-effect unification metrics for one tissue.

        Args:
            gen_adata: Generated cells (fixed ``batch_id``; ``cell_type_coarse``
                carried via obs passthrough).
            real_adata: Real cells (many ``batch_id``; ``cell_type_coarse``).
            output_dir: Directory for the UMAP overlay PNGs.
            device: ``"cuda"`` (rapids_singlecell) or ``"cpu"``.

        Returns:
            Dict of metric name → scalar (NaN where a dep/column is missing).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, a in (("real", real_adata), ("gen", gen_adata)):
            if "batch_id" not in a.obs:
                raise ValueError(f"batch_effect: {name}_adata missing obs 'batch_id'.")
        has_celltype = (
            CELL_TYPE_KEY in real_adata.obs.columns
            and CELL_TYPE_KEY in gen_adata.obs.columns
        )
        if not has_celltype:
            print(f"  WARNING: '{CELL_TYPE_KEY}' missing on gen/real; cLISI skipped.")

        n_batches = int(real_adata.obs["batch_id"].nunique())
        metrics: dict[str, Any] = {
            "n_real": int(real_adata.n_obs),
            "n_gen": int(gen_adata.n_obs),
            "n_real_batches": n_batches,
            "gen_batch_id": str(gen_adata.obs["batch_id"].iloc[0]),
        }

        # --- A: real, raw + Harmony ---
        print("  [A] embedding real (raw + Harmony)...")
        t0 = time.time()
        A = real_adata.copy()
        harmony_ok = _embed(
            A, device=device, batch_key="batch_id", with_harmony=(n_batches >= 2)
        )
        metrics["ilisi_batch_id__real_raw"] = _lisi(
            A, rep="X_pca", key="batch_id", kind="ilisi"
        )
        if has_celltype:
            metrics["clisi_celltype__real_raw"] = _lisi(
                A, rep="X_pca", key=CELL_TYPE_KEY, kind="clisi"
            )
        if harmony_ok:
            metrics["ilisi_batch_id__real_harmony"] = _lisi(
                A, rep="X_pca_harmony", key="batch_id", kind="ilisi"
            )
            if has_celltype:
                metrics["clisi_celltype__real_harmony"] = _lisi(
                    A, rep="X_pca_harmony", key=CELL_TYPE_KEY, kind="clisi"
                )
        else:
            metrics["ilisi_batch_id__real_harmony"] = float("nan")
        print(f"    A done ({time.time()-t0:.1f}s)")

        # --- A∪C: real + generated (fixed batch) ---
        print("  [A∪C] embedding real + generated...")
        t0 = time.time()
        real_c = real_adata.copy()
        gen_c = gen_adata.copy()
        real_c.obs["data_source"] = "real"
        gen_c.obs["data_source"] = "gen"
        AC = ad.concat([real_c, gen_c], join="inner")
        AC.obs_names_make_unique()
        _embed(AC, device=device, batch_key="batch_id", with_harmony=False)

        metrics["ilisi_batch_id__real_plus_gen"] = _lisi(
            AC, rep="X_pca", key="batch_id", kind="ilisi"
        )
        metrics["ilisi_data_source__real_plus_gen"] = _lisi(
            AC, rep="X_pca", key="data_source", kind="ilisi"
        )
        if has_celltype:
            metrics["clisi_celltype__real_plus_gen"] = _lisi(
                AC, rep="X_pca", key=CELL_TYPE_KEY, kind="clisi"
            )
        print(f"    A∪C done ({time.time()-t0:.1f}s)")

        # --- UMAP overlay of the A∪C joint embedding ---
        try:
            self._save_umap(AC, output_dir, device=device, has_celltype=has_celltype)
        except Exception as exc:  # noqa: BLE001 - plotting must not fail the metric
            print(f"  WARNING: UMAP overlay failed: {exc}")

        return metrics

    def _save_umap(
        self, AC: ad.AnnData, output_dir: Path, *, device: str, has_celltype: bool
    ) -> None:
        """Compute one joint UMAP of A∪C and save overlays by the key colorings.

        Uses a single shared embedding (coordinates comparable across cells) and
        colors it by data_source / batch_id / cell_type — the standard
        batch-integration figure. ``_embed`` already ran (X_pca present); here we
        only build neighbors + UMAP on that rep.

        Args:
            AC: Joint real+gen AnnData (embedded; has ``data_source``/``batch_id``).
            output_dir: Directory for the PNGs.
            device: ``"cuda"`` or ``"cpu"``.
            has_celltype: Whether a cell-type coloring is available.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if device == "cuda":
            try:
                import rapids_singlecell as rsc

                rsc.get.anndata_to_GPU(AC)
                rsc.pp.neighbors(AC, use_rep="X_pca", n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
                rsc.tl.umap(AC)
                rsc.get.anndata_to_CPU(AC)
            except Exception as exc:  # noqa: BLE001
                print(f"  GPU UMAP failed ({exc}); using scanpy CPU.")
                try:
                    import rapids_singlecell as rsc

                    rsc.get.anndata_to_CPU(AC)  # ensure X is back on CPU
                except Exception:  # noqa: BLE001
                    pass
                sc.pp.neighbors(AC, use_rep="X_pca", n_neighbors=N_NEIGHBORS)
                sc.tl.umap(AC)
        else:
            sc.pp.neighbors(AC, use_rep="X_pca", n_neighbors=N_NEIGHBORS)
            sc.tl.umap(AC)

        colorings = ["data_source", "batch_id"] + ([CELL_TYPE_KEY] if has_celltype else [])
        for color in colorings:
            legend = "right margin" if AC.obs[color].nunique() <= 20 else None
            fig = sc.pl.umap(
                AC, color=color, show=False, return_fig=True,
                size=6, alpha=0.6 if AC.n_obs > 20000 else 1.0,
                title=f"A∪C — {color}", legend_loc=legend,
            )
            fig.savefig(str(output_dir / f"umap_{color}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved UMAP overlays ({', '.join(colorings)}) -> {output_dir}")
