"""Convert gene var_names in h5ad files between Ensembl IDs and gene names.

Reusable utility for gene ID ↔ name conversion. The conversion logic is
extracted from ``merge_with_control.py`` for standalone use.

Supports two modes:
- ``name_to_id``: gene name → Ensembl ID (e.g. SAMD11 → ENSG00000187634)
- ``id_to_name``: Ensembl ID → gene name (e.g. ENSG00000187634 → SAMD11)

Can be used as a CLI tool or imported as a library function.

CLI usage::

    # Convert in-place (overwrite input file)
    python -m workflows.utils.convert_gene_ids \\
        --input /path/to/merged.h5ad \\
        --mapping data/vcc/gene_id_name.csv \\
        --direction name_to_id

    # Convert to a new file
    python -m workflows.utils.convert_gene_ids \\
        --input /path/to/merged.h5ad \\
        --output /path/to/converted.h5ad \\
        --mapping data/vcc/gene_id_name.csv \\
        --direction name_to_id

Library usage::

    from workflows.utils.convert_gene_ids import convert_var_names
    adata = convert_var_names(adata, "data/vcc/gene_id_name.csv", direction="name_to_id")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
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
        their original value. Original names are backed up in
        ``adata.var["gene_name"]`` or ``adata.var["ensembl_id"]``.
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
        raise ValueError(f"Unknown direction: {direction!r}. Use 'name_to_id' or 'id_to_name'.")

    original = adata.var_names.tolist()
    new = [mapping.get(v, v) for v in original]
    mapped = sum(1 for o, n in zip(original, new) if o != n)
    print(f"  var_names converted ({direction}): {mapped}/{len(original)} mapped")

    adata.var[backup_col] = original
    adata.var_names = new
    adata.var_names_make_unique()
    return adata


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert gene var_names in h5ad between Ensembl IDs and gene names.",
    )
    parser.add_argument("--input", required=True, help="Input h5ad file.")
    parser.add_argument("--output", default=None,
                        help="Output h5ad file. If not given, overwrites input.")
    parser.add_argument("--mapping", required=True,
                        help="Tab-separated gene ID/name mapping file (no header).")
    parser.add_argument("--direction", required=True, choices=["name_to_id", "id_to_name"],
                        help="Conversion direction.")
    args = parser.parse_args()

    output = args.output or args.input

    print(f"Loading: {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  Shape: {adata.shape}")
    print(f"  var_names sample: {adata.var_names[:3].tolist()}")

    adata = convert_var_names(adata, args.mapping, direction=args.direction)

    print(f"Saving: {output}")
    adata.write_h5ad(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
