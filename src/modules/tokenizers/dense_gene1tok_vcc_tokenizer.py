from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse

import anndata
import pandas as pd

from ..common import TokenSequence
from .base import Tokenizer


class DenseGene1TokVCCTokenizer(Tokenizer):
    """VCC dense gene1tok tokenizer for single-cell gene expression sequences.

    Encodes one AnnData row into a fixed-length token sequence composed of a
    condition prefix followed by one count token per gene.  The condition
    prefix has length ``condition_prefix_length`` (default 5) and encodes
    cell type and target gene.  The expression body has length
    ``max_gene_count`` and encodes UMI counts for each gene in
    ``gene_names_ordered`` order.

    Legacy counterpart:
    - ``archive/legacy/lit_gpt/sc_tokenizer_vcc_dense_gene1tok.py``
      ``VCCDenseGene1TokSingleCellTokenizer``

    Attributes:
        vocab_size: Total token ids (special + count + cell-type + target-gene).
        seq_length: Fixed sequence length = ``condition_prefix_length + max_gene_count``.
        eos_id: End-of-sequence token id (``</s>``).
        condition_prefix_length: Number of leading condition tokens (default 5).
        max_gene_count: Number of gene positions in the expression body.
        gene_name_to_dense_idx: Map from gene name to dense position index.
        gene_names_ordered: Ordered tuple of gene names defining the dense layout.
        required_obs_keys: Obs column names that must be present (or defaulted)
            for encoding: ``("cell_type", "target_gene")``.
        default_obs_values: Default string values used when an obs key is absent.
    """

    def __init__(
        self,
        *,
        vocab_file: str,
        condition_schema: str,
        condition_prefix_length: int = 5,
        condition_cell_type_idx: int = 1,
        condition_target_gene_high_idx: int = 2,
        condition_target_gene_low_idx: int = 3,
        default_cell_type_value: str = "H1",
    ) -> None:
        """Initialize the VCC dense gene1tok tokenizer.

        Resolves and loads the vocab file, then populates all encoding/decoding
        lookup tables via ``_load_vocab_metadata()``.

        Args:
            vocab_file: Path to the tokenizer JSON vocab file.  Resolved
                relative to the repo root when not an absolute path; also
                tried under ``archive/legacy/`` for backward compatibility.
            condition_schema: Condition schema name shared with the dataset
                module (e.g. ``"vcc_celltype_target_gene"``).  Used as a
                stable identifier; does not affect encoding logic directly.
            condition_prefix_length: Number of token slots reserved for the
                condition prefix at the start of every encoded sequence.
                Default is 5 (the VCC fixed layout).
            condition_cell_type_idx: Index within the condition prefix where
                the cell-type token is placed.  Default is 1.
            condition_target_gene_high_idx: Index within the condition prefix
                where the target-gene token is placed.  The ``high_idx``
                name is a legacy compatibility alias; gene1tok VCC encodes
                only one semantic target-gene token.  Default is 2.
            condition_target_gene_low_idx: Index within the condition prefix
                reserved for the legacy second target-gene placeholder.
                Filled with ``<cond>`` in the current implementation.
                Default is 3.
            default_cell_type_value: Cell type string used as a fallback when
                the obs column is absent or empty (e.g. ``"H1"``).
        """
        # Resolve vocab file path, falling back to the archive/legacy subtree
        resolved_vocab_file = Path(vocab_file)
        if not resolved_vocab_file.is_file():
            repo_root_candidate = Path(__file__).resolve().parents[3]
            legacy_candidate = repo_root_candidate / "archive" / "legacy" / vocab_file
            if legacy_candidate.is_file():
                resolved_vocab_file = legacy_candidate
            else:
                raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_file}")

        self.vocab_file = str(resolved_vocab_file)
        self.condition_schema = condition_schema
        self.condition_prefix_length = condition_prefix_length
        self.condition_cell_type_idx = condition_cell_type_idx
        # These legacy names are intentionally retained as compatibility-only
        # slot identifiers for the fixed 5-token condition prefix.
        self.condition_target_gene_high_idx = condition_target_gene_high_idx
        self.condition_target_gene_low_idx = condition_target_gene_low_idx
        self.default_cell_type_value = default_cell_type_value
        # Fixed obs key names used when reading condition values from AnnData
        self.condition_cell_type_obs_key = "cell_type"
        self.condition_target_obs_key = "target_gene"
        self.condition_target_is_gene = True
        self._load_vocab_metadata()

    @property
    def vocab_size(self) -> int:
        """Total number of distinct token ids across all vocab groups.

        Computed as: special tokens + count tokens + cell-type tokens +
        special target-gene tokens + per-gene target tokens.

        Returns:
            Integer vocabulary size populated by ``_load_vocab_metadata()``.
        """
        return self._vocab_size

    def _load_vocab_metadata(self) -> None:
        """Parse the vocab JSON file and populate all encoding/decoding tables.

        Reads the five top-level sections of the vocab file and builds the
        lookup structures required by ``encode``, ``encode_condition``, and
        ``decode``.  Also computes derived scalar fields (``seq_length``,
        ``vocab_size``, ``default_obs_values``, ``required_obs_keys``).

        Raises:
            KeyError: If a required section or key is missing from the vocab JSON.
            json.JSONDecodeError: If the vocab file is not valid JSON.
        """
        # Load the five top-level sections from the vocab JSON
        with open(self.vocab_file, "r", encoding="utf-8") as handle:
            vocab_data = json.load(handle)

        self.special_token_vocab = dict(vocab_data["special_token_vocab"])
        self.count_to_token_id_info = dict(vocab_data["count_to_token_id_info"])
        self.cell_type_token_vocab = dict(vocab_data["cell_type_token_vocab"])
        self.target_gene_to_token_id_info = dict(vocab_data["target_gene_to_token_id_info"])
        self.special_target_gene_vocab = dict(vocab_data["special_target_gene_vocab"])

        # Unpack special token ids
        self.eos_id = int(self.special_token_vocab["</s>"])
        self.cond_id = int(self.special_token_vocab["<cond>"])
        self.unknown_gene_id = int(self.special_token_vocab["<unk_gene>"])

        # Unpack count encoding boundaries and mapping rules
        self.overflow_token_id = int(self.count_to_token_id_info["overflow_token_id"])
        self.min_count_token_id = int(self.count_to_token_id_info["min_token_id"])
        self.max_count_token_id = int(self.count_to_token_id_info["max_token_id"])
        self.count_mapping_rules = tuple(self.count_to_token_id_info["count_mapping_rules"])

        # Unpack the non-targeting (control) target-gene token; first key in the dict
        self.non_targeting_word = str(next(iter(self.special_target_gene_vocab)))
        self.non_targeting_id = int(self.special_target_gene_vocab[self.non_targeting_word])

        # Build reverse lookup: token id -> cell type string for decode
        self.token_id_to_cell_type = {
            int(token_id): cell_type for cell_type, token_id in self.cell_type_token_vocab.items()
        }

        # Unpack target-gene token offset and gene name ordering
        self.target_gene_token_id_offset = int(self.target_gene_to_token_id_info["target_gene_token_id_offset"])
        self.max_gene_count = int(self.target_gene_to_token_id_info["max_gene_count"])
        self.gene_names_ordered = tuple(self.target_gene_to_token_id_info["gene_names_ordered"])

        # Build forward lookup: gene name -> dense position index
        self.gene_name_to_dense_idx = {
            gene_name: idx for idx, gene_name in enumerate(self.gene_names_ordered)
        }

        # Derived sequence shape and compatibility fields
        self.seq_length = self.condition_prefix_length + self.max_gene_count
        self.target_gene_2token_width = None  # unused in gene1tok; retained for compat

        # Default and required obs values used by dataset fallback logic
        self.default_obs_values = {
            self.condition_cell_type_obs_key: self.default_cell_type_value,
            self.condition_target_obs_key: self.non_targeting_word,
        }
        self.required_obs_keys = (
            self.condition_cell_type_obs_key,
            self.condition_target_obs_key,
        )

        # Compute total vocabulary size across all token groups
        self._vocab_size = (
            len(self.special_token_vocab)
            + (self.max_count_token_id - self.min_count_token_id + 1)
            + len(self.cell_type_token_vocab)
            + len(self.special_target_gene_vocab)
            + self.max_gene_count
        )

    def encode(
        self,
        adata: Any,
        row_index: int = 0,
        *,
        gene_names: tuple[str, ...] | None = None,
    ) -> TokenSequence:
        """Encode one AnnData row into a fixed-length token sequence.

        Reads condition values from ``adata.obs.iloc[row_index]`` and gene
        expression counts from ``adata.X[row_index]``.  The returned sequence
        has length ``seq_length`` = ``condition_prefix_length + max_gene_count``.

        Condition fallback order:
        - ``cell_type``: obs column ``"cell_type"``; falls back to
          ``default_cell_type_value`` if absent or empty.
        - ``target_gene``: obs column ``"target_gene"``; falls back to
          ``"gene_target"`` column, then to ``non_targeting_word`` if both
          are absent or empty.

        Args:
            adata: ``anndata.AnnData`` object containing the sample(s).
            row_index: Zero-based row index within *adata* to encode.
            gene_names: Optional pre-resolved gene name tuple aligned with
                the columns of ``adata.X``.  When ``None``, derived from
                ``adata.var_names``.  Providing this avoids repeated
                ``var_names`` lookups when encoding many rows from the same
                AnnData.

        Returns:
            A ``TokenSequence`` with ``token_ids`` of length ``seq_length``
            and ``condition_prefix_length`` set to ``self.condition_prefix_length``.
        """
        obs = adata.obs.iloc[row_index]

        # Read cell type with fallback to default
        cell_type = str(obs.get("cell_type", self.default_cell_type_value) or self.default_cell_type_value)

        # Read target gene with fallback to legacy key then non-targeting word
        target_gene_val = obs.get("target_gene")
        if target_gene_val in (None, ""):
            target_gene_val = obs.get("gene_target", self.non_targeting_word)
        target_gene = str(target_gene_val or self.non_targeting_word)

        prefix_tokens = self.encode_condition({"cell_type": cell_type, "target_gene": target_gene})

        # Resolve gene names from var_names if not provided by caller
        if gene_names is None:
            gene_names = tuple(str(g) for g in adata.var_names.tolist())

        dense_tokens = self._encode_dense_counts(row=adata.X[row_index], gene_names=gene_names)
        return TokenSequence(
            token_ids=tuple(prefix_tokens + dense_tokens),
            condition_prefix_length=self.condition_prefix_length,
        )

    def encode_condition(self, conditions: Mapping[str, str]) -> list[int]:
        """Encode condition values into a fixed-length prefix token list.

        Produces exactly ``condition_prefix_length`` (= 5) token ids using the
        VCC fixed layout:

        - Index 0: ``<cond>`` sentinel.
        - Index 1 (``condition_cell_type_idx``): cell-type token id.
        - Index 2 (``condition_target_gene_high_idx``): target-gene token id.
        - Index 3 (``condition_target_gene_low_idx``): ``<cond>`` placeholder
          (legacy second target-gene slot, unused in gene1tok).
        - Index 4: ``<cond>`` sentinel.

        Args:
            conditions: Mapping with string values for the following keys:
                - ``"cell_type"``: cell type label (e.g. ``"H1"``).  Falls
                  back to ``default_cell_type_value`` if absent or not in
                  the vocabulary.
                - ``"target_gene"``: target gene name (e.g. ``"BRCA1"``),
                  the non-targeting control word (e.g. ``"non-targeting"``),
                  or ``"<unk_gene>"``.  Falls back to ``non_targeting_word``
                  if absent.

        Returns:
            List of ``condition_prefix_length`` token ids.
        """
        cell_type = str(conditions.get("cell_type", self.default_cell_type_value))
        target_gene = str(conditions.get("target_gene", self.non_targeting_word))
        return [
            self.cond_id,
            int(self.cell_type_token_vocab.get(cell_type, self.cell_type_token_vocab[self.default_cell_type_value])),
            self._encode_target_gene_token(target_gene),
            self.cond_id,
            self.cond_id,
        ]

    def _encode_dense_counts(self, *, row: Any, gene_names: Sequence[str]) -> list[int]:
        """Encode one gene-expression row into a dense list of count tokens.

        Produces a list of length ``max_gene_count`` where position ``i``
        corresponds to ``gene_names_ordered[i]``.  Genes present in *row*
        but absent from ``gene_name_to_dense_idx`` are silently skipped.

        Args:
            row: Gene-expression data for one cell.  Accepts a scipy sparse
                matrix row (any format) or a dense array-like of shape
                ``(n_genes,)`` or ``(1, n_genes)``.
            gene_names: Sequence of gene name strings aligned with the
                columns of *row*.  Length must equal the number of columns.

        Returns:
            List of ``max_gene_count`` token ids, one per gene in
            ``gene_names_ordered`` order.  Genes with zero count get the
            zero-count token id.
        """
        zero_id = self._encode_count_token(0)
        tokens = [zero_id] * self.max_gene_count

        # Sparse path: only iterate over non-zero entries
        if scipy.sparse.issparse(row):
            csr = row.tocsr()
            for local_idx, count in zip(csr.indices.tolist(), csr.data.tolist()):
                gene_name = str(gene_names[local_idx])
                dense_idx = self.gene_name_to_dense_idx.get(gene_name)
                if dense_idx is not None:
                    tokens[dense_idx] = self._encode_count_token(int(count))
            return tokens

        # Dense path: iterate all columns, skip zeros
        row_array = np.asarray(row).reshape(-1)
        for local_idx, count in enumerate(row_array.tolist()):
            if int(count) <= 0:
                continue
            gene_name = str(gene_names[local_idx])
            dense_idx = self.gene_name_to_dense_idx.get(gene_name)
            if dense_idx is not None:
                tokens[dense_idx] = self._encode_count_token(int(count))
        return tokens

    def decode(self, sequence: TokenSequence) -> anndata.AnnData:
        """Decode a token sequence back into a 1-row AnnData object.

        Reconstructs condition values from the condition prefix and gene
        expression counts from the expression body.  The result has the
        canonical VCC output layout:

        - ``X``: sparse CSR matrix of shape ``(1, max_gene_count)`` with
          decoded integer UMI counts.
        - ``obs``: DataFrame with ``cell_type`` and ``target_gene`` columns.
        - ``var``: DataFrame indexed by ``gene_names_ordered``.

        Args:
            sequence: A ``TokenSequence`` with at least
                ``condition_prefix_length`` token ids.  Token ids beyond
                ``condition_prefix_length + max_gene_count`` are ignored.

        Returns:
            A 1-row ``anndata.AnnData``.

        Raises:
            ValueError: If ``sequence`` has fewer than
                ``condition_prefix_length`` token ids.
        """
        token_ids = list(sequence.token_ids)
        if len(token_ids) < self.condition_prefix_length:
            raise ValueError(
                f"Token sequence too short: expected at least {self.condition_prefix_length}, got {len(token_ids)}."
            )

        # Decode condition prefix: cell type and target gene
        cell_type_token = int(token_ids[self.condition_cell_type_idx])
        target_gene_token = int(token_ids[self.condition_target_gene_high_idx])
        obs = {
            self.condition_cell_type_obs_key: self.token_id_to_cell_type.get(
                cell_type_token, self.default_cell_type_value,
            ),
            self.condition_target_obs_key: self._decode_target_gene_token(target_gene_token),
        }

        # Decode expression body: collect non-zero (gene_idx, count) pairs
        count_tokens = token_ids[self.condition_prefix_length:]
        zero_id = self._encode_count_token(0)
        gene_indices: list[int] = []
        counts: list[int] = []
        for i, tid in enumerate(count_tokens[: self.max_gene_count]):
            if tid == zero_id:
                continue
            count = self._decode_count_token(tid)
            if count > 0:
                gene_indices.append(i)
                counts.append(count)

        # Build sparse expression matrix and return AnnData
        X = scipy.sparse.csr_matrix(
            (counts, ([0] * len(counts), gene_indices)),
            shape=(1, self.max_gene_count),
            dtype=np.int32,
        )
        return anndata.AnnData(
            X=X,
            obs=pd.DataFrame(obs, index=[0]),
            var=pd.DataFrame(index=pd.Index(list(self.gene_names_ordered))),
        )

    def _encode_target_gene_token(self, gene_name: str) -> int:
        """Encode a target-gene name to its token id.

        Lookup priority:
        1. ``"<unk_gene>"`` -> ``unknown_gene_id``.
        2. Case-insensitive match to ``non_targeting_word`` -> ``non_targeting_id``.
        3. Exact match in ``gene_name_to_dense_idx`` -> ``target_gene_token_id_offset + dense_idx``.
        4. Otherwise -> ``unknown_gene_id``.

        Args:
            gene_name: Target gene name string, the non-targeting control
                word, or ``"<unk_gene>"``.

        Returns:
            Token id for the given gene name.
        """
        normalized_gene_name = str(gene_name)
        if normalized_gene_name == "<unk_gene>":
            return self.unknown_gene_id
        if normalized_gene_name.lower() == self.non_targeting_word.lower():
            return self.non_targeting_id
        dense_idx = self.gene_name_to_dense_idx.get(normalized_gene_name)
        if dense_idx is None:
            return self.unknown_gene_id
        return self.target_gene_token_id_offset + dense_idx

    def _decode_target_gene_token(self, token_id: int) -> str:
        """Decode a target-gene token id back to a gene name string.

        Reverse of ``_encode_target_gene_token``.

        Args:
            token_id: Token id produced by ``_encode_target_gene_token``.

        Returns:
            Gene name string, ``non_targeting_word``, or ``"<unk_gene>"``
            if the token id is out of range.
        """
        if token_id == self.unknown_gene_id:
            return "<unk_gene>"
        if token_id == self.non_targeting_id:
            return self.non_targeting_word
        dense_idx = token_id - self.target_gene_token_id_offset
        if 0 <= dense_idx < len(self.gene_names_ordered):
            return self.gene_names_ordered[dense_idx]
        return "<unk_gene>"

    def _encode_count_token(self, count: int) -> int:
        """Encode an integer UMI count to its token id.

        Clamps non-positive values to zero before delegating to the
        rule-based encoder.

        Args:
            count: Raw integer UMI count.  Values <= 0 are treated as 0.

        Returns:
            Token id for the given count value.
        """
        if count <= 0:
            return self._encode_count_token_from_rules(0)
        return self._encode_count_token_from_rules(int(count))

    def _encode_count_token_from_rules(self, count: int) -> int:
        """Map an integer count to a token id using the vocab count mapping rules.

        Applies the ordered list of ``count_mapping_rules`` from the vocab JSON.
        Each rule covers a contiguous range of counts and specifies one of two
        mapping methods:

        - ``"identity"``: direct linear mapping; token id = ``min_token_id``
          + (count - ``min_count``).
        - ``"2_important_digits"``: lossy compression retaining the two most
          significant digits; token id encodes the digit-length bucket and the
          leading two-digit value.

        Boundary handling:
        - count < ``min_count``: returns ``min_count_token_id``.
        - count > ``max_count``: returns ``overflow_token_id``.

        Args:
            count: Non-negative integer count already clamped by
                ``_encode_count_token``.

        Returns:
            Token id for the given count.

        Raises:
            ValueError: If an unsupported ``mapping_method`` is encountered,
                or if no rule covers the count despite being in range.
        """
        min_count = int(self.count_to_token_id_info["min_count"])
        max_count = int(self.count_to_token_id_info["max_count"])

        # Handle boundary counts outside the rule range
        if count < min_count:
            return self.min_count_token_id
        if count > max_count:
            return self.overflow_token_id

        # Find the matching rule and apply its mapping method
        for rule in self.count_mapping_rules:
            rule_min = int(rule["min_count"])
            rule_max = int(rule["max_count"])
            if rule_min <= count <= rule_max:
                base_token = int(rule["min_token_id"])
                mapping_method = str(rule["mapping_method"])
                if mapping_method == "identity":
                    return base_token + (count - rule_min)
                if mapping_method == "2_important_digits":
                    length = len(str(count))
                    divisor = 10 ** (length - 2)
                    return base_token + (length - 3) * 90 + (count // divisor - 10)
                raise ValueError(f"Unsupported count mapping method: {mapping_method}")
        raise ValueError(f"Count {count} is not covered by tokenizer count mapping rules.")

    def _decode_count_token(self, token_id: int) -> int:
        """Decode a count token id back to an integer count.

        Reverses ``_encode_count_token_from_rules``.  For
        ``"2_important_digits"`` rules the returned value is the
        reconstructed representative count (the lower bound of the encoded
        bucket), not necessarily the original exact count.

        Args:
            token_id: Token id produced by ``_encode_count_token``.

        Returns:
            Decoded integer count, or 0 if no rule matches the token id.
        """
        # Overflow token maps to the vocabulary maximum count
        if token_id == self.overflow_token_id:
            return int(self.count_to_token_id_info["max_count"])

        # Walk rules to find the one whose token range contains this token id
        for rule in self.count_mapping_rules:
            rule_min_tid = int(rule["min_token_id"])
            rule_max_tid = int(rule["max_token_id"])
            if rule_min_tid <= token_id <= rule_max_tid:
                mapping_method = str(rule["mapping_method"])
                if mapping_method == "identity":
                    return int(rule["min_count"]) + (token_id - rule_min_tid)
                if mapping_method == "2_important_digits":
                    offset = token_id - rule_min_tid
                    length = offset // 90 + 3
                    divisor = 10 ** (length - 2)
                    return (offset - (length - 3) * 90 + 10) * divisor
        return 0
