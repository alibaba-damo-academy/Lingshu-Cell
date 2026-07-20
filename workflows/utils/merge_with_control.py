"""Merge generated predictions with control cells.

External evaluation tools (e.g. ``cell-eval``) require the predict h5ad
to contain both perturbed cells AND control (non-targeting / PBS) cells
so the tool can compute predicted differential expression internally.

The infer workflow only produces perturbed cells; this utility appends
control cells from a control h5ad (which may be a pre-extracted control
file or a full training set — only matching control cells are used).

Pipeline:
    1. Load predicted h5ad (output of infer workflow).
    2. Optionally convert predicted var_names (Ensembl ↔ gene name).
    3. Load control h5ad, filter to control cells (target_obs_key == control_value).
    4. Optionally restrict control by a secondary obs (e.g. donor for PARSE).
    5. Optionally convert control var_names.
    6. Align predicted var_names to control var_names (drop extras,
       error on missing).
    7. Concatenate control + predicted, save to output.

Usage::

    python -m workflows.utils.merge_with_control \\
        --predict /path/to/sample_dir/merged.h5ad \\
        --control /path/to/adata_Control.h5ad \\
        --output /path/to/sample_dir/merged_with_control.h5ad

Reference: archive/legacy/scripts_after_vcc_zh/data_merge/merge_h5ad_and_control.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd


def convert_var_names(
    adata: ad.AnnData,
    mapping_file: str,
    *,
    direction: str,
) -> ad.AnnData:
    """Convert ``var_names`` between Ensembl IDs and gene names.

    Args:
        adata: Input AnnData.
        mapping_file: Tab-separated file with two columns (no header):
            column 1 = Ensembl ID, column 2 = gene name.
        direction: ``"name_to_id"`` (gene name → Ensembl) or
            ``"id_to_name"`` (Ensembl → gene name).

    Returns:
        AnnData with converted ``var_names``. Unmapped entries keep
        their original value. Original names are backed up to ``adata.var``.
    """
    df = pd.read_csv(mapping_file, sep="\t", header=None, names=["ensembl_id", "gene_name"])

    if direction == "name_to_id":
        df = df.drop_duplicates(subset=["gene_name"], keep="first")
        mapping = dict(zip(df["gene_name"], df["ensembl_id"]))
        backup_col = "gene_name"
    elif direction == "id_to_name":
        mapping = dict(zip(df["ensembl_id"], df["gene_name"]))
        backup_col = "ensembl_id"
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    original = adata.var_names.tolist()
    new = [mapping.get(v, v) for v in original]
    mapped = sum(1 for o, n in zip(original, new) if o != n)
    print(f"  var_names converted ({direction}): {mapped}/{len(original)} mapped")

    adata.var[backup_col] = original
    adata.var_names = new
    adata.var_names_make_unique()
    return adata


def filter_control_cells(
    control_h5ad: str,
    *,
    target_obs_key: str,
    control_value: str,
    cell_type_obs_key: Optional[str] = None,
    cell_type_filter: Optional[str] = None,
) -> ad.AnnData:
    """Load a h5ad and extract only the control cells.

    The input file may be a full training set or a pre-filtered control
    h5ad — in either case only cells matching the control criteria are
    returned.

    Args:
        control_h5ad: Path to h5ad file containing control cells.
        target_obs_key: Column name identifying perturbation target
            (e.g. ``"target_gene"`` for VCC, ``"cytokine"`` for PARSE).
        control_value: Value identifying control cells in ``target_obs_key``
            (e.g. ``"non-targeting"`` for VCC, ``"PBS"`` for PARSE).
        cell_type_obs_key: Optional secondary filter column (e.g. donor).
        cell_type_filter: Optional secondary filter value.

    Returns:
        AnnData with only the matching control cells.
    """
    print(f"Loading control h5ad: {control_h5ad}")
    adata = ad.read_h5ad(control_h5ad)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    if target_obs_key not in adata.obs.columns:
        raise ValueError(
            f"Control h5ad missing required obs column {target_obs_key!r}. "
            f"Available: {list(adata.obs.columns)}"
        )

    mask = adata.obs[target_obs_key].astype(str) == str(control_value)
    control = adata[mask].copy()
    print(f"  Control cells ({target_obs_key}={control_value!r}): {control.n_obs}")

    if cell_type_obs_key and cell_type_filter:
        if cell_type_obs_key not in control.obs.columns:
            raise ValueError(
                f"Control h5ad missing required obs column {cell_type_obs_key!r}."
            )
        mask = control.obs[cell_type_obs_key].astype(str) == str(cell_type_filter)
        control = control[mask].copy()
        print(
            f"  After {cell_type_obs_key}={cell_type_filter!r} filter: "
            f"{control.n_obs} cells"
        )

    if control.n_obs == 0:
        raise ValueError(
            f"No control cells found. Check target_obs_key={target_obs_key!r}, "
            f"control_value={control_value!r}."
        )
    return control


def align_var_names(adata: ad.AnnData, reference_var_names: pd.Index) -> ad.AnnData:
    """Align ``adata.var_names`` to a reference index.

    Drops any var in ``adata`` not present in the reference; errors if any
    reference var is missing from ``adata``.

    Args:
        adata: Source AnnData to align.
        reference_var_names: Target var_names (e.g. control's var_names).

    Returns:
        ``adata`` with var ordered and filtered to match ``reference_var_names``.
    """
    ref = pd.Index(reference_var_names).astype(str)
    cur = pd.Index(adata.var_names).astype(str)
    missing = ref.difference(cur)
    extra = cur.difference(ref)

    print(
        f"  Aligning genes: current={len(cur)}, reference={len(ref)}, "
        f"extra_to_drop={len(extra)}, missing_from_predict={len(missing)}"
    )

    if len(missing) > 0:
        preview = ", ".join(missing[:10].tolist())
        raise ValueError(
            f"Reference has {len(missing)} genes not in predict (first 10: {preview}). "
            f"Predict h5ad must contain at least all reference genes."
        )

    return adata[:, ref].copy()


def _combine_obs_columns(
    adata: ad.AnnData,
    *,
    columns: list[str],
    target_key: str,
    control_value: str,
) -> None:
    """Combine multiple obs columns into a single perturbation label column.

    Joins the values of ``columns`` with ``'+'``.  When all constituent
    values equal ``control_value``, the combined value is set to
    ``control_value`` directly (avoiding ``'ctrl+ctrl'`` style labels).
    Source columns that differ from ``target_key`` are dropped.

    The operation is performed **in-place** on ``adata.obs``.

    Args:
        adata: AnnData whose ``obs`` contains the columns to combine.
        columns: Ordered list of obs column names to combine.
        target_key: Name of the resulting combined column (typically the
            first element of ``columns``).
        control_value: Value identifying control cells; rows where all
            constituent columns equal this are assigned ``control_value``
            instead of a ``'+'``-joined string.

    Raises:
        ValueError: If any column in ``columns`` is missing from
            ``adata.obs``.
    """
    missing = [c for c in columns if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found in predict obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    parts = [adata.obs[c].astype(str) for c in columns]

    # Build combined label per row: drop parts equal to control_value
    # (treated as no-op slots), then '+'-join the remaining non-control
    # parts.  If all parts are control_value, the label is control_value.
    # This matches the source data convention where single-gene cells have
    # a single gene label and control cells have just the control label.
    def _join_non_control(row_parts: list[str]) -> str:
        non_ctrl = [p for p in row_parts if p != control_value]
        return "+".join(non_ctrl) if non_ctrl else control_value

    parts_matrix = np.array([p.values for p in parts]).T  # (n_rows, n_cols)
    combined_vals = [_join_non_control(list(row)) for row in parts_matrix]
    combined = pd.Series(combined_vals, index=adata.obs.index)

    adata.obs[target_key] = combined

    # Drop source columns that are not the target key
    drop_cols = [c for c in columns if c != target_key]
    if drop_cols:
        adata.obs.drop(columns=drop_cols, inplace=True)

    n_control = int((combined == control_value).sum())
    n_pert = len(adata) - n_control
    print(
        f"  Combined columns {columns} -> {target_key!r}: "
        f"{n_pert} perturbation + {n_control} control cells, "
        f"{adata.obs[target_key].nunique()} unique labels"
    )


def merge_with_control(
    predict_h5ad: str,
    control_h5ad: str,
    output_path: str,
    *,
    target_obs_key: str = "target_gene",
    control_value: str = "non-targeting",
    cell_type_obs_key: Optional[str] = None,
    cell_type_filter: Optional[str] = None,
    gene_id_name_csv: Optional[str] = None,
    convert_predict: str = "none",
    convert_control: str = "none",
    combine_columns: Optional[list[str]] = None,
    combine_control_value: Optional[str] = None,
) -> None:
    """Append control cells to predicted h5ad.

    Args:
        predict_h5ad: Input merged predictions (output of infer workflow).
        control_h5ad: Source of control cells (control-only file or full
            training set; only matching control cells are used).
        output_path: Output path for combined h5ad.
        target_obs_key: Obs column identifying perturbation target.
        control_value: Value identifying control cells in the control
            h5ad (used for filtering).
        cell_type_obs_key: Optional secondary filter column.
        cell_type_filter: Optional secondary filter value.
        gene_id_name_csv: Optional Ensembl ↔ gene name mapping (TSV).
        convert_predict: ``"none"`` (default), ``"name_to_id"``, or
            ``"id_to_name"`` — convert predict var_names before merge.
        convert_control: same semantics applied to control cells.
        combine_columns: Optional list of obs column names to combine
            into the first column's name using ``'+'`` as separator.
            The combined column replaces that column in the predict obs.
            When all constituent values equal ``combine_control_value``
            (or ``control_value`` if not set), the combined value is set
            to that value (no ``'+'`` join).  Source columns other than
            the first are dropped after combining.
            Example: ``["target_gene", "target_gene_2"]`` produces
            ``'GENE1+GENE2'`` for double-gene perturbation labels.
        combine_control_value: Value identifying control rows when
            combining columns.  When all constituent columns equal this
            value, the combined label is set to this value instead of
            joining with ``'+'``.  Defaults to ``control_value`` if not
            specified.  This is separate from ``control_value`` because
            the control h5ad may use a different column/value for
            filtering (e.g. ``is_control==True``) than the predict h5ad
            uses for perturbation labels (e.g.
            ``target_gene==non-perturbation``).
    """
    # --- Load predictions ---
    print(f"Loading predictions: {predict_h5ad}")
    predict = ad.read_h5ad(predict_h5ad)
    print(f"  {predict.n_obs} cells x {predict.n_vars} genes")

    # --- Optional column combination (before any other processing) ---
    if combine_columns:
        combine_ctrl = combine_control_value if combine_control_value else control_value
        _combine_obs_columns(
            predict,
            columns=combine_columns,
            target_key=combine_columns[0],
            control_value=combine_ctrl,
        )

    # --- Optional conversion on predict ---
    if convert_predict != "none":
        if not gene_id_name_csv:
            raise ValueError("--gene_id_name_csv required when --convert_predict is set.")
        predict = convert_var_names(predict, gene_id_name_csv, direction=convert_predict)

    # --- Load and filter control cells ---
    control = filter_control_cells(
        control_h5ad,
        target_obs_key=target_obs_key,
        control_value=control_value,
        cell_type_obs_key=cell_type_obs_key,
        cell_type_filter=cell_type_filter,
    )

    # --- Optional conversion on control ---
    if convert_control != "none":
        if not gene_id_name_csv:
            raise ValueError("--gene_id_name_csv required when --convert_control is set.")
        control = convert_var_names(control, gene_id_name_csv, direction=convert_control)

    # --- Align predict var_names to control var_names (control is reference) ---
    predict = align_var_names(predict, control.var_names)

    # --- Concatenate control + predict ---
    print(f"Concatenating: control={control.n_obs} + predict={predict.n_obs}")
    combined = ad.concat([control, predict], join="outer", index_unique=None)
    combined.obs.index = [f"cell_{i}" for i in range(combined.n_obs)]
    print(f"Combined shape: {combined.shape}")

    # --- Save ---
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(str(output))
    print(f"Saved: {output}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge predicted h5ad with control cells.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--predict", required=True, help="Input merged predictions h5ad.")
    parser.add_argument("--control", required=True,
                        help="h5ad containing control cells (control-only or full training set).")
    parser.add_argument("--output", required=True, help="Output combined h5ad path.")
    parser.add_argument(
        "--target_obs_key", default="target_gene",
        help="Obs column for perturbation target (default: target_gene).",
    )
    parser.add_argument(
        "--control_value", default="non-targeting",
        help="Value identifying control cells (default: non-targeting).",
    )
    parser.add_argument(
        "--cell_type_obs_key", default="",
        help="Optional secondary filter column (e.g. donor for PARSE).",
    )
    parser.add_argument(
        "--cell_type_filter", default="",
        help="Optional secondary filter value.",
    )
    parser.add_argument(
        "--gene_id_name_csv", default="",
        help="TSV with Ensembl ↔ gene name mapping (no header).",
    )
    parser.add_argument(
        "--convert_predict", default="none",
        choices=["none", "name_to_id", "id_to_name"],
        help="Convert predict var_names before merge (default: none).",
    )
    parser.add_argument(
        "--convert_control", default="none",
        choices=["none", "name_to_id", "id_to_name"],
        help="Convert control var_names before merge (default: none).",
    )
    parser.add_argument(
        "--combine_columns", default="",
        help=(
            "Comma-separated obs column names to combine into the first "
            "column's name using '+' separator (e.g. 'target_gene,target_gene_2'). "
            "Control rows (all columns == combine_control_value) keep that value. "
            "Source columns other than the first are dropped."
        ),
    )
    parser.add_argument(
        "--combine_control_value", default="",
        help=(
            "Value identifying control rows when combining columns. "
            "When all constituent columns equal this value, the combined "
            "label is set to this value instead of '+'-joining. "
            "Defaults to --control_value if not specified."
        ),
    )
    args = parser.parse_args()

    # Parse combine_columns: comma-separated string -> list or None
    combine_cols = None
    if args.combine_columns:
        combine_cols = [c.strip() for c in args.combine_columns.split(",") if c.strip()]

    merge_with_control(
        predict_h5ad=args.predict,
        control_h5ad=args.control,
        output_path=args.output,
        target_obs_key=args.target_obs_key,
        control_value=args.control_value,
        cell_type_obs_key=args.cell_type_obs_key or None,
        cell_type_filter=args.cell_type_filter or None,
        gene_id_name_csv=args.gene_id_name_csv or None,
        convert_predict=args.convert_predict,
        convert_control=args.convert_control,
        combine_columns=combine_cols,
        combine_control_value=args.combine_control_value or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
