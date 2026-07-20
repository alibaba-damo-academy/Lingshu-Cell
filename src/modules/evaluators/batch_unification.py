"""Batch-unification metric for the foundation eval (Track B, Tahoe plates).

This is the test the batch-effect work actually wanted: does the model treat
``batch_id`` as a removable **technical knob**? It exploits the one substrate in
the pretrain corpus where biology ⊥ batch — Tahoe's DMSO controls, where the same
cell lines recur on every plate (cell-village design), so ``cell_line_id`` (the
biology axis) and ``plate`` / ``batch_id`` (the technical axis) are independent.
On CxG this is impossible (cell_description bakes in donor/disease, correlating
with batch) — that substrate gets the separate fidelity/integration eval
(``batch_integration``, Track A).

Design (see ``.docs/pretrain/batch_effect_eval_design_v1.html``):

- **B1 (real only)** — scIB baseline/ceiling on real DMSO controls:
  ``iLISI(plate)`` raw vs Harmony (low→high = plate effect present and removable)
  and ``cLISI(cell_line)`` (should stay high = biology preserved).
- **B2a (generated only — the counterfactual)** — generate each cell line under
  all 14 plate batch_ids (biology fixed, only ``batch_id`` varies). ``iLISI(plate)``
  high in *raw* gen = the model collapsed the technical axis (plate is a knob it
  did not entangle with biology); ``cLISI(cell_line)`` high = cell-line identity
  preserved across plates.
- **B2b (real + generated)** — co-embed, Harmony on ``batch_id``: ``iLISI(data_source)``
  (does gen match real after de-batching), ``iLISI(plate)``, ``cLISI(cell_line)``.
- **B2c (plate-direction fidelity, bonus)** — does generated (cell_line, plateK)
  reproduce the *real* per-plate shift (not just ignore batch)? Cosine between the
  real and generated plate-effect vectors + nearest-real-centroid accuracy. This
  guards against "high iLISI just because the model ignores batch_id".

Score convention (scib ``me`` API, ``scale=True``): iLISI and cLISI both in
[0, 1], higher=better. GPU via rapids_singlecell with a scanpy CPU fallback;
shared embed/LISI helpers reused from ``batch_integration``. Run in the
``rapids_sc`` env. Registered as ``batch_unification``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from . import register_metric
from .batch_integration import _embed, _ensure_cpu, _lisi, N_NEIGHBORS, N_PCS

CELL_LINE_KEY = "cell_line_id"
BATCH_KEY = "batch_id"  # Tahoe obs batch_id is already ``tahoe_plateK``


def _save_umap(
    AC: ad.AnnData, output_dir: Path, *, device: str, rep: str, tag: str, colorings: list
) -> None:
    """Compute one joint UMAP on ``rep`` and save overlays by each coloring.

    Args:
        AC: Joint real+gen AnnData (already embedded; has ``obsm[rep]``).
        output_dir: Directory for the PNGs.
        device: ``"cuda"`` or ``"cpu"``.
        rep: ``obsm`` key to build neighbors+UMAP on.
        tag: Filename tag (``"corrected"`` / ``"raw"``).
        colorings: obs columns to color the shared embedding by.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    for color in colorings:
        legend = "right margin" if AC.obs[color].nunique() <= 25 else None
        fig = sc.pl.umap(
            AC, color=color, show=False, return_fig=True,
            size=6, alpha=0.6 if AC.n_obs > 20000 else 1.0,
            title=f"real∪gen ({tag}) — {color}", legend_loc=legend,
        )
        fig.savefig(str(output_dir / f"umap_{tag}_{color}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved UMAP overlays [{tag}] ({', '.join(colorings)}) -> {output_dir}")


def _plate_direction_fidelity(AC: ad.AnnData, *, rep: str = "X_pca") -> dict[str, float]:
    """Does generated (cell_line, plate) reproduce the real per-plate shift?

    In the joint embedding ``obsm[rep]``, for each cell line compute the per-plate
    centroid for real and for gen, subtract the cell line's own mean to get a
    "plate-effect" vector per plate, then compare real vs gen:

    - ``plate_dir_cosine``: mean cosine similarity between the real and gen
      plate-effect vectors (averaged over cell line × plate). High = gen moves
      cells in the same technical direction the real plate does.
    - ``plate_nn_accuracy``: fraction of (cell_line, plate) where the gen plate
      centroid's nearest real plate centroid (same cell line) is the matching
      plate. High = gen's per-plate signature is plate-specific, not random.

    Guards against a degenerate "high iLISI just because gen ignores batch_id".

    Args:
        AC: Joint real+gen AnnData with ``obs[data_source/batch_id/cell_line_id]``
            and ``obsm[rep]``.
        rep: Embedding to measure in (default raw joint PCA).

    Returns:
        Dict with ``plate_dir_cosine`` and ``plate_nn_accuracy`` (NaN if not
        computable).
    """
    X = np.asarray(AC.obsm[rep])
    obs = AC.obs
    ds = obs["data_source"].to_numpy()
    cl = obs[CELL_LINE_KEY].astype(str).to_numpy()
    pl = obs[BATCH_KEY].astype(str).to_numpy()

    def _centroids(mask) -> dict:
        """Map (cell_line, plate) -> mean embedding for the masked subset."""
        out: dict = {}
        for c in np.unique(cl[mask]):
            for p in np.unique(pl[mask & (cl == c)]):
                sel = mask & (cl == c) & (pl == p)
                if sel.sum() > 0:
                    out[(c, p)] = X[sel].mean(axis=0)
        return out

    real_c = _centroids(ds == "real")
    gen_c = _centroids(ds == "gen")

    cosines: list[float] = []
    nn_hits = 0
    nn_total = 0
    for c in np.unique(cl):
        plates = sorted({p for (cc, p) in real_c if cc == c} & {p for (cc, p) in gen_c if cc == c})
        if len(plates) < 2:
            continue
        real_mean = np.mean([real_c[(c, p)] for p in plates], axis=0)
        gen_mean = np.mean([gen_c[(c, p)] for p in plates], axis=0)
        # Plate-effect direction agreement.
        for p in plates:
            dr = real_c[(c, p)] - real_mean
            dg = gen_c[(c, p)] - gen_mean
            nr, ng = np.linalg.norm(dr), np.linalg.norm(dg)
            if nr > 1e-8 and ng > 1e-8:
                cosines.append(float(np.dot(dr, dg) / (nr * ng)))
        # Nearest real-plate-centroid accuracy for each gen plate centroid.
        real_mat = np.stack([real_c[(c, p)] for p in plates])
        for i, p in enumerate(plates):
            d = np.linalg.norm(real_mat - gen_c[(c, p)], axis=1)
            nn_total += 1
            if plates[int(np.argmin(d))] == p:
                nn_hits += 1

    return {
        "plate_dir_cosine": float(np.mean(cosines)) if cosines else float("nan"),
        "plate_nn_accuracy": (nn_hits / nn_total) if nn_total else float("nan"),
    }


@register_metric("batch_unification")
class BatchUnificationMetrics:
    """Tahoe DMSO batch-unification metrics (B1 real / B2 gen counterfactual)."""

    name = "batch_unification"

    def compute(
        self,
        *,
        gen_adata: ad.AnnData,
        real_adata: ad.AnnData,
        output_dir: Path,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute Track-B batch-unification metrics + UMAP overlays.

        Args:
            gen_adata: Generated DMSO cells; obs must carry ``batch_id`` (plate)
                and ``cell_line_id`` via the infer obs passthrough.
            real_adata: Real DMSO cells with ``batch_id`` / ``cell_line_id``.
            output_dir: Directory for the UMAP overlay PNGs.
            device: ``"cuda"`` (rapids_singlecell) or ``"cpu"``.

        Returns:
            Dict of metric name → scalar (NaN where a dep/column is missing).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Validate obs contract ---
        for label, a in (("real", real_adata), ("gen", gen_adata)):
            for col in (BATCH_KEY, CELL_LINE_KEY):
                if col not in a.obs:
                    raise ValueError(f"batch_unification: {label}_adata missing obs '{col}'.")

        n_plates = int(real_adata.obs[BATCH_KEY].nunique())
        n_lines = int(real_adata.obs[CELL_LINE_KEY].nunique())
        metrics: dict[str, Any] = {
            "n_real": int(real_adata.n_obs),
            "n_gen": int(gen_adata.n_obs),
            "n_plates": n_plates,
            "n_cell_lines": n_lines,
        }

        # ---------- B1: real only (plate effect baseline + Harmony ceiling) ----------
        print("  [B1] embedding real DMSO (raw + Harmony)...")
        t0 = time.time()
        B = real_adata.copy()
        harmony_ok = _embed(B, device=device, batch_key=BATCH_KEY, with_harmony=(n_plates >= 2))
        metrics["b1_ilisi_plate__raw"] = _lisi(B, rep="X_pca", key=BATCH_KEY, kind="ilisi")
        metrics["b1_clisi_cellline__raw"] = _lisi(B, rep="X_pca", key=CELL_LINE_KEY, kind="clisi")
        if harmony_ok:
            metrics["b1_ilisi_plate__harmony"] = _lisi(B, rep="X_pca_harmony", key=BATCH_KEY, kind="ilisi")
            metrics["b1_clisi_cellline__harmony"] = _lisi(B, rep="X_pca_harmony", key=CELL_LINE_KEY, kind="clisi")
        else:
            metrics["b1_ilisi_plate__harmony"] = float("nan")
        print(f"    B1 done ({time.time()-t0:.1f}s)")

        # ---------- B2a: generated only (the counterfactual) ----------
        # Same cell line under all 14 plates: if plate is a pure technical knob the
        # model collapsed, the plate versions mix (high raw iLISI(plate)) while cell
        # lines stay apart (high cLISI(cell_line)).
        print("  [B2a] embedding generated DMSO (counterfactual)...")
        t0 = time.time()
        G = gen_adata.copy()
        gen_harmony_ok = _embed(G, device=device, batch_key=BATCH_KEY, with_harmony=(n_plates >= 2))
        metrics["b2_gen_ilisi_plate__raw"] = _lisi(G, rep="X_pca", key=BATCH_KEY, kind="ilisi")
        metrics["b2_gen_clisi_cellline__raw"] = _lisi(G, rep="X_pca", key=CELL_LINE_KEY, kind="clisi")
        if gen_harmony_ok:
            metrics["b2_gen_ilisi_plate__harmony"] = _lisi(G, rep="X_pca_harmony", key=BATCH_KEY, kind="ilisi")
        print(f"    B2a done ({time.time()-t0:.1f}s)")

        # ---------- B2b: real + generated (integration / fidelity) ----------
        print("  [B2b] embedding real + generated (Harmony on plate)...")
        t0 = time.time()
        real_c = real_adata.copy()
        gen_c = gen_adata.copy()
        real_c.obs["data_source"] = "real"
        gen_c.obs["data_source"] = "gen"
        AC = ad.concat([real_c, gen_c], join="inner")
        AC.obs_names_make_unique()
        ac_harmony_ok = _embed(AC, device=device, batch_key=BATCH_KEY, with_harmony=(n_plates >= 2))

        metrics["b2_ilisi_datasource__raw"] = _lisi(AC, rep="X_pca", key="data_source", kind="ilisi")
        metrics["b2_ilisi_plate__raw"] = _lisi(AC, rep="X_pca", key=BATCH_KEY, kind="ilisi")
        metrics["b2_clisi_cellline__raw"] = _lisi(AC, rep="X_pca", key=CELL_LINE_KEY, kind="clisi")
        if ac_harmony_ok:
            metrics["b2_ilisi_datasource__corrected"] = _lisi(AC, rep="X_pca_harmony", key="data_source", kind="ilisi")
            metrics["b2_ilisi_plate__corrected"] = _lisi(AC, rep="X_pca_harmony", key=BATCH_KEY, kind="ilisi")
            metrics["b2_clisi_cellline__corrected"] = _lisi(AC, rep="X_pca_harmony", key=CELL_LINE_KEY, kind="clisi")
        else:
            metrics["b2_ilisi_datasource__corrected"] = float("nan")
        print(f"    B2b done ({time.time()-t0:.1f}s)")

        # ---------- B2c: plate-direction fidelity (bonus; never fail the metric) ----------
        try:
            fid = _plate_direction_fidelity(AC, rep="X_pca")
            metrics.update(fid)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: plate-direction fidelity failed: {exc}")

        # ---------- UMAP overlays (corrected + raw) ----------
        try:
            colorings = [BATCH_KEY, CELL_LINE_KEY, "data_source"]
            if ac_harmony_ok:
                _save_umap(AC, output_dir, device=device, rep="X_pca_harmony",
                           tag="corrected", colorings=colorings)
            _save_umap(AC, output_dir, device=device, rep="X_pca", tag="raw", colorings=colorings)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: UMAP overlay failed: {exc}")

        return metrics
