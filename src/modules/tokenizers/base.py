from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..common.interfaces import TokenSequence


class Tokenizer(Protocol):
    """Minimal tokenizer protocol for the train/inference workflow.

    Every concrete tokenizer (``DenseGene1TokTokenizer``,
    ``DenseGene1TokVCCTokenizer``) must satisfy this interface.  The
    workflow interacts with the tokenizer exclusively through these
    attributes and methods.

    Attributes:
        vocab_size: Total number of distinct token ids, including special
            tokens, count tokens, categorical tokens, and gene-index tokens.
        seq_length: Fixed output sequence length produced by ``encode()``
            (= ``condition_prefix_length + max_gene_count``).
        eos_id: Token id for the end-of-sequence ``</s>`` token, used
            for padding shorter sequences.
        condition_prefix_length: Number of leading tokens reserved for
            condition encoding (e.g. 5 for the VCC layout).
        max_gene_count: Number of gene expression positions in the
            token sequence (one token per gene).
        gene_name_to_dense_idx: Mapping from gene name strings to their
            dense positional index in the expression portion of the
            sequence.
    """

    vocab_size: int
    seq_length: int
    eos_id: int
    condition_prefix_length: int
    max_gene_count: int
    gene_name_to_dense_idx: Mapping[str, int]

    def encode(self, adata: Any, row_index: int = 0, *, gene_names: Any = None) -> TokenSequence:
        """Encode one AnnData sample into a complete token sequence.

        Reads condition values from ``adata.obs.iloc[row_index]`` and
        gene expression from ``adata.X[row_index]``.

        Args:
            adata: An ``anndata.AnnData`` object containing the sample(s).
            row_index: Row index within *adata* to encode.
            gene_names: Optional pre-resolved gene name tuple.  When
                ``None``, derived from ``adata.var_names``.

        Returns:
            A ``TokenSequence`` with ``token_ids`` of length
            ``seq_length`` (condition prefix + expression tokens).
        """

    def encode_condition(self, conditions: Mapping[str, str]) -> list[int]:
        """Encode condition values into a prefix token list.

        Used by ``encode()`` internally, and directly by inference
        workflows that need to construct condition prefixes without
        expression data.

        Args:
            conditions: Mapping from slot ``obs_key`` to its string
                value.  All slots must be present (e.g.
                ``{"cell_type": "H1", "target_gene": "GENE_A"}``).

        Returns:
            List of ``condition_prefix_length`` token ids.
        """

    def decode(self, sequence: TokenSequence) -> Any:
        """Decode one generated token sequence back into an AnnData.

        Reconstructs a 1-row ``anndata.AnnData`` with:

        - ``X``: sparse CSR matrix ``(1, max_gene_count)`` of decoded
          expression counts.
        - ``obs``: DataFrame with condition columns (e.g. ``cell_type``,
          ``target_gene``).
        - ``var``: DataFrame indexed by ``gene_names_ordered``.

        Args:
            sequence: A ``TokenSequence`` with at least
                ``condition_prefix_length`` token ids.

        Returns:
            A 1-row ``anndata.AnnData``.
        """
