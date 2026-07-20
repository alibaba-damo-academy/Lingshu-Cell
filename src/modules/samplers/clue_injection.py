"""Clue-based biology prior injection for guided diffusion sampling.

At inference time, pre-fills known differentially expressed gene counts
("clues") into the initially masked sequence before the diffusion
sampling loop begins.  This constrains generation toward biologically
realistic expression profiles for perturbation experiments.

Ported from the authoritative legacy implementation at
``archive/legacy/eval/utils_vcc.py`` (``extract_clues``).

Typical usage (VCC gene perturbation)::

    clues_dict = extract_clues(
        target_genes=["BRCA2", "TP53"],
        clues_df=pd.read_csv("whole_dataset_stringent_clue.csv"),
        gene_ensembl_to_name={"ENSG00000139618": "BRCA2", ...},
        filter_rule_sets=["reference_mean>=2,target_mean<=1,cell_type!=H1"],
        count_rules=["count_1"],
    )

    apply_clue_prefill(
        x0=x0,
        target_gene="BRCA2",
        clues_dict=clues_dict,
        tokenizer=tokenizer,
        condition_prefix_length=5,
        condition_target_count=1,
        clue_overwrite_target=True,
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Gene ID mapping helpers
# ---------------------------------------------------------------------------


def load_gene_id_name_mapping(csv_path: str | Path) -> Dict[str, str]:
    """Load gene Ensembl ID -> gene name mapping from a tab-separated CSV.

    The file has no header and two columns: ``ensembl_id<TAB>gene_name``.

    Args:
        csv_path: Path to the gene ID mapping file (e.g.
            ``data/vcc/gene_id_name.csv``).

    Returns:
        Dictionary mapping Ensembl IDs to gene names.
    """
    df = pd.read_csv(csv_path, sep="\t", header=None, names=["ensembl_id", "gene_name"])
    return dict(zip(df["ensembl_id"], df["gene_name"]))


def _unify_target_name(
    target_name: str,
    gene_name_to_ensembl: Dict[str, str],
    use_ensembl: bool = True,
) -> str:
    """Convert a target gene name to the clue CSV's target identifier.

    For VCC experiments, the clue CSV uses Ensembl IDs as target identifiers,
    while the condition CSV uses gene names.  For PARSE experiments, the
    cytokine name is used directly.

    Args:
        target_name: Target gene/perturbation name from the condition CSV.
        gene_name_to_ensembl: Reverse lookup from gene name to Ensembl ID.
        use_ensembl: If ``True``, convert gene name to Ensembl ID (VCC).
            If ``False``, use ``target_name`` directly (PARSE).

    Returns:
        Unified target name for matching against the clue CSV.

    Raises:
        KeyError: If ``use_ensembl`` is True and ``target_name`` is not
            found in the mapping.
    """
    if use_ensembl:
        if target_name not in gene_name_to_ensembl:
            raise KeyError(
                f"Target gene name '{target_name}' not found in "
                f"gene_name_to_ensembl mapping."
            )
        return gene_name_to_ensembl[target_name]
    return target_name


# ---------------------------------------------------------------------------
# Clue extraction
# ---------------------------------------------------------------------------


def extract_clues(
    target_genes: List[str],
    clues_df: pd.DataFrame,
    gene_ensembl_to_name: Dict[str, str],
    filter_rule_sets: Optional[List[str]] = None,
    count_rules: Optional[List[str]] = None,
    gathering_mode: str = "any",
    group_col: str = "cell_type",
    use_ensembl_as_target: bool = True,
    clue_reference_means: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Tuple[str, Optional[int]]]]:
    """Extract clue genes and their expression counts for each target gene.

    This is a clean reimplementation of the legacy ``extract_clues`` from
    ``archive/legacy/eval/utils_vcc.py``.

    Args:
        target_genes: Target gene/perturbation names from the condition CSV
            (e.g. ``["BRCA2", "TP53"]`` for VCC, or ``["IFN-beta"]`` for
            PARSE).
        clues_df: DataFrame loaded from the clue CSV with columns:
            ``target``, ``feature``, ``<group_col>``, ``target_mean``,
            ``reference_mean``, ``log2_fold_change``.
        gene_ensembl_to_name: Mapping from Ensembl IDs to gene names.
        filter_rule_sets: List of filtering condition strings.  Rules
            within a string are AND-ed (comma-separated); different
            strings in the list are OR-ed.  Example:
            ``["reference_mean>=2,target_mean<=1,cell_type!=H1"]``.
            If ``None``, no filtering is applied and all rows are kept.
        count_rules: List of count assignment methods, one per rule set.
            Supported values:

            - ``"count_<n>"`` (e.g. ``"count_0"``, ``"count_1"``): fixed
              integer count.
            - ``"count_by_round_target_mean"``: ``round(mean(target_mean))``
              across cell type groups.
            - ``"min_abs_log2_fold_change"``: count derived from
              ``ref_mean * 2^log2fc`` using the minimum absolute fold change.
              Requires ``clue_reference_means``.
            - ``"no_count"``: returns ``None`` (position noted but no
              count placed).

            Defaults to ``["count_1"]`` if not specified.
        gathering_mode: ``"any"`` or ``"all"``.  Controls whether a
            feature gene must appear in at least one (``any``) or all
            (``all``) cell type groups for a given target.
        group_col: Column name in the clue CSV for grouping (``"cell_type"``
            for VCC, ``"DonorID"`` for PARSE).
        use_ensembl_as_target: If ``True``, convert target gene names
            to Ensembl IDs for matching against the clue CSV (VCC).
            If ``False``, use target names directly (PARSE).
        clue_reference_means: Optional mapping from gene Ensembl ID to
            reference mean expression.  Required only for the
            ``"min_abs_log2_fold_change"`` count rule.

    Returns:
        Dictionary mapping each target gene name (original form) to a
        sorted, deduplicated list of ``(feature_gene_name, count)`` tuples.
        The count is ``None`` for ``"no_count"`` rules.
    """
    if count_rules is None:
        count_rules = ["count_1"]

    if filter_rule_sets is not None and len(filter_rule_sets) != len(count_rules):
        raise ValueError(
            f"filter_rule_sets length ({len(filter_rule_sets)}) must equal "
            f"count_rules length ({len(count_rules)})."
        )

    # Build reverse mapping: gene_name -> ensembl_id
    gene_name_to_ensembl = {v: k for k, v in gene_ensembl_to_name.items()}

    # --- Step 1: Convert target gene names to unified identifiers ---
    unified_targets: Dict[str, str] = {}  # unified_name -> original_name
    for tg in target_genes:
        try:
            unified = _unify_target_name(tg, gene_name_to_ensembl, use_ensembl_as_target)
            unified_targets[unified] = tg
        except KeyError:
            # Target gene not in mapping; skip silently
            pass

    clue_target_set = set(clues_df["target"].unique())
    intersecting = set(unified_targets.keys()) & clue_target_set

    if not intersecting:
        print("No overlapping target genes found between task list and clue CSV.")
        return {}

    print(f"Found {len(intersecting)} overlapping target genes for clues.")

    # --- Step 2: Filter sub-table ---
    sub_table = clues_df[clues_df["target"].isin(intersecting)].copy()

    if filter_rule_sets:
        filtered_chunks = []
        for rule_set_str, count_rule in zip(filter_rule_sets, count_rules):
            query_parts = []
            for rule in rule_set_str.split(","):
                rule = rule.strip()
                match = re.match(r"([a-zA-Z0-9_]+)\s*(==|!=|>=|<=|>|<)\s*(.+)", rule)
                if not match:
                    raise ValueError(f"Invalid filter rule format: '{rule}'")
                col, op, val = match.groups()
                try:
                    float(val)
                    query_parts.append(f"`{col}` {op} {val}")
                except ValueError:
                    query_parts.append(f"`{col}` {op} '{val.strip()}'")

            query_string = " & ".join(f"({p})" for p in query_parts)
            chunk = sub_table.query(query_string).copy()
            if not chunk.empty:
                chunk["_count_rule_"] = count_rule
                filtered_chunks.append(chunk)

        if filtered_chunks:
            sub_table = pd.concat(filtered_chunks, ignore_index=True)
        else:
            sub_table = pd.DataFrame(columns=sub_table.columns.tolist() + ["_count_rule_"])
    else:
        sub_table["_count_rule_"] = count_rules[0]

    if sub_table.empty:
        print("Sub-table empty after filtering. No clues to process.")
        return {}

    # --- Step 3: Process each target gene ---
    clues_dict: Dict[str, List[Tuple[str, Optional[int]]]] = {}

    for target_id in tqdm(intersecting, desc="Extracting clues"):
        target_df = sub_table[sub_table["target"] == target_id].copy()
        if target_df.empty:
            continue

        # Filter out genes with inconsistent regulation direction
        target_df["direction"] = np.sign(
            target_df["target_mean"] - target_df["reference_mean"]
        )
        inconsistent = target_df.groupby("feature")["direction"].transform(
            lambda x: x.min() < 0 and x.max() > 0
        )
        target_df = target_df[~inconsistent]
        if target_df.empty:
            continue

        # Gathering mode: filter features across cell type groups
        if gathering_mode == "all":
            n_groups = target_df[group_col].nunique()
            if n_groups == 0:
                continue
            feat_counts = target_df.groupby("feature")[group_col].nunique()
            features = feat_counts[feat_counts == n_groups].index.tolist()
        elif gathering_mode == "any":
            features = target_df["feature"].unique().tolist()
        else:
            raise ValueError(f"Invalid gathering_mode: '{gathering_mode}'")

        if not features:
            continue

        # Calculate count for each feature
        feature_expr_list: List[Tuple[str, Optional[int]]] = []
        for feature_id in features:
            if feature_id not in gene_ensembl_to_name:
                continue
            feature_name = gene_ensembl_to_name[feature_id]
            rows = target_df[target_df["feature"] == feature_id]

            for count_rule, rule_rows in rows.groupby("_count_rule_"):
                final_count = _compute_count(
                    count_rule, rule_rows, feature_id, clue_reference_means
                )
                feature_expr_list.append((feature_name, final_count))

        if feature_expr_list:
            original_name = unified_targets[target_id]
            clues_dict[original_name] = sorted(set(feature_expr_list))

    print(f"Extracted clues for {len(clues_dict)} target genes.")
    return clues_dict


def _compute_count(
    count_rule: str,
    rows: pd.DataFrame,
    feature_id: str,
    reference_means: Optional[Dict[str, float]],
) -> Optional[int]:
    """Compute the expression count for a clue gene under a given rule.

    Args:
        count_rule: The count assignment method string.
        rows: DataFrame rows for this feature under the current rule.
        feature_id: Ensembl ID of the feature gene.
        reference_means: Optional reference mean expression mapping.

    Returns:
        Integer count, or ``None`` for ``"no_count"`` rule.

    Raises:
        ValueError: If the count rule is unsupported or if
            ``"min_abs_log2_fold_change"`` is used without reference means.
    """
    if count_rule == "no_count":
        return None

    if count_rule == "count_by_round_target_mean":
        return round(float(np.mean(rows["target_mean"].values)))

    count_match = re.match(r"count_(\d+)", count_rule)
    if count_match:
        return int(count_match.group(1))

    if count_rule == "min_abs_log2_fold_change":
        if reference_means is None:
            raise ValueError(
                "'min_abs_log2_fold_change' count rule requires "
                "clue_reference_means to be provided."
            )
        idx = rows["log2_fold_change"].abs().idxmin()
        log2fc = rows.loc[idx, "log2_fold_change"]
        ref_mean = reference_means[feature_id]
        return max(1, round(ref_mean * (2 ** log2fc)))

    raise ValueError(f"Unsupported count rule: '{count_rule}'")


# ---------------------------------------------------------------------------
# Sequence pre-filling
# ---------------------------------------------------------------------------


def apply_clue_prefill(
    x0: torch.Tensor,
    target_gene: str,
    clues_dict: Dict[str, List[Tuple[str, Optional[int]]]],
    tokenizer: object,
    condition_prefix_length: int,
    condition_target_count: int = -1,
    clue_overwrite_target: bool = True,
) -> int:
    """Pre-fill clue gene positions in the initial masked sequence.

    Modifies ``x0`` in-place by placing known expression count tokens at
    the clue gene positions (replacing ``mask_token_id``).  Also optionally
    places the target gene's own count.

    Args:
        x0: Initial sequence tensor ``(B, L)`` with mask tokens in the
            gene expression region.  Modified in-place.
        target_gene: Name of the perturbation target gene for this task.
        clues_dict: Output of ``extract_clues``; mapping from target
            gene names to lists of ``(feature_gene_name, count)`` tuples.
        tokenizer: Tokenizer instance with ``gene_name_to_dense_idx``
            attribute and ``_encode_count_token(count)`` method.
        condition_prefix_length: Number of prefix tokens before the gene
            expression region.
        condition_target_count: Count value to place at the target gene
            position.  ``-1`` means "do not place" (leave masked).
        clue_overwrite_target: If ``False``, skip clue placement when the
            clue gene is the same as the target gene and
            ``condition_target_count >= 0``.

    Returns:
        Number of positions actually pre-filled (including target gene
        if applicable).
    """
    gene_name_to_idx = tokenizer.gene_name_to_dense_idx
    encode_count = tokenizer._encode_count_token
    placed = 0

    # --- Place target gene count ---
    if condition_target_count >= 0 and target_gene in gene_name_to_idx:
        dense_idx = gene_name_to_idx[target_gene]
        pos = condition_prefix_length + dense_idx
        token_id = encode_count(condition_target_count)
        x0[:, pos] = token_id
        placed += 1

    # --- Place clue genes ---
    if target_gene not in clues_dict:
        return placed

    for gene_name, count in clues_dict[target_gene]:
        if gene_name not in gene_name_to_idx:
            continue

        # Skip target gene if overwrite is disabled
        if (
            not clue_overwrite_target
            and gene_name == target_gene
            and condition_target_count >= 0
        ):
            continue

        if count is not None:
            dense_idx = gene_name_to_idx[gene_name]
            pos = condition_prefix_length + dense_idx
            token_id = encode_count(count)
            x0[:, pos] = token_id
            placed += 1

    return placed
