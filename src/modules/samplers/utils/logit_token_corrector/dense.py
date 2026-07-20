"""Dense logit corrector for batch-aware reverse diffusion sampling.

Operates on full-sequence logits of shape ``(B, L, D)`` for dense
tokenizers where each position after the condition prefix encodes a
single gene's expression count.

Ported from the authoritative ``BatchDenseLogitAndTokenCorrector`` in
``TransMDM_VCC/eval/logit_token_corrector.py`` (L352-391).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class DenseLogitCorrector:
    """Batch-compatible logit corrector for dense (count-only) tokenizers.

    Strategies:
        ``default``:
            Positions in the gene-expression region
            ``[condition_prefix_length, seq_length)`` can only sample
            count tokens — all non-count logits are set to ``-inf``.
        ``restrict_valid_genes``:
            For gene positions NOT in the supplied allow-list, force
            all *non-zero* count logits to ``-inf`` so only count=0
            can be sampled.

    Args:
        seq_shape: ``(B, L)`` — batch size and padded sequence length.
        logit_dim: Vocabulary size (``D``), excluding the mask token.
        tokenizer: Dense tokenizer instance. Must expose
            ``min_count_token_id``, ``max_count_token_id``,
            ``condition_prefix_length``, ``seq_length``, and optionally
            ``gene_name_to_dense_idx`` / ``_encode_count`` for
            ``restrict_valid_genes``.
        strategies: Comma-separated strategy string.
        sample_gene_names: Gene names to allow when using
            ``restrict_valid_genes``.
        device: Torch device.
    """

    def __init__(
        self,
        seq_shape: Tuple[int, int],
        logit_dim: int,
        tokenizer: object,
        strategies: str = "default",
        sample_gene_names: Optional[List[str]] = None,
        device: str = "cuda",
        **extra_kwargs: object,
    ) -> None:
        parsed = strategies.split(",")
        for s in parsed:
            assert s in ("", "default", "restrict_valid_genes"), f"Unknown strategy: {s!r}"
        self.strategies = parsed
        self._is_noop = self.strategies == [""]
        if self._is_noop:
            return

        self.B = seq_shape[0]
        self.L = seq_shape[1]
        self.D = logit_dim
        self.mask_id = logit_dim
        self.device = device

        _min = getattr(tokenizer, "min_count_token_id", None)
        if _min is None:
            _min = getattr(tokenizer, "min_value_token_id", None)
        self.min_count_token_id = _min

        _max = getattr(tokenizer, "max_count_token_id", None)
        if _max is None:
            _max = getattr(tokenizer, "max_value_token_id", None)
        self.max_count_token_id = _max
        self.condition_prefix_length = int(tokenizer.condition_prefix_length)
        self.seq_length = int(tokenizer.seq_length)

        logit_indices = torch.arange(logit_dim, device=device)
        self.positions = (
            torch.arange(self.L, dtype=torch.long, device=device)
            .reshape(1, -1)
            .repeat(self.B, 1)
        )

        # ``count_mask[i]`` is True for logit indices that are NOT valid count tokens.
        self.count_mask = ~(
            (logit_indices >= self.min_count_token_id)
            & (logit_indices <= self.max_count_token_id)
        )
        self.non_count_cols = torch.where(self.count_mask)[0]

        self.allowed_gene_dense_mask: Optional[torch.Tensor] = None
        self.nonzero_count_token_ids: Optional[torch.Tensor] = None
        if "restrict_valid_genes" in self.strategies:
            if sample_gene_names is not None and len(sample_gene_names) > 0:
                if not hasattr(tokenizer, "gene_name_to_dense_idx"):
                    raise ValueError(
                        "Tokenizer must have gene_name_to_dense_idx for restrict_valid_genes."
                    )
                gene_count = int(self.seq_length - self.condition_prefix_length)
                allowed_mask = torch.zeros(gene_count, dtype=torch.bool, device=device)
                for gene_name in sample_gene_names:
                    dense_idx = tokenizer.gene_name_to_dense_idx[gene_name]
                    allowed_mask[int(dense_idx)] = True
                self.allowed_gene_dense_mask = allowed_mask

                zero_count_token_id = int(tokenizer._encode_count(0))
                nonzero_count_token_mask = (
                    (logit_indices >= self.min_count_token_id)
                    & (logit_indices <= self.max_count_token_id)
                    & (logit_indices != zero_count_token_id)
                )
                self.nonzero_count_token_ids = torch.where(nonzero_count_token_mask)[0]

    def correct_logits(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        *,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Correct logits in-place for a batch of sequences.

        Args:
            logits: ``(B, L, D)`` full-position logits when
                ``masked_positions`` is None, or ``(N_mask, D)`` masked-only
                logits when ``masked_positions`` is provided.
            tokens: ``(B, L)`` — current token ids (masked and unmasked).
            masked_positions: Optional ``(N_mask,)`` int tensor giving the
                original sequence position for each row in ``logits``. When
                provided, logits is treated as a flat ``(N_mask, D)`` tensor.

        Returns:
            The corrected ``logits`` tensor (modified in-place).
        """
        if self._is_noop:
            return logits

        if masked_positions is not None:
            return self._correct_logits_masked_only(logits, masked_positions)

        in_gene_region = (
            (self.positions >= self.condition_prefix_length)
            & (self.positions < self.seq_length)
        )

        if "default" in self.strategies:
            correction_mask = in_gene_region.unsqueeze(-1) & self.count_mask.view(1, 1, -1)
            logits[correction_mask] = -torch.inf

        if "restrict_valid_genes" in self.strategies and self.allowed_gene_dense_mask is not None:
            dense_idx = self.positions - self.condition_prefix_length
            clamped_idx = dense_idx.clamp(min=0, max=self.allowed_gene_dense_mask.shape[0] - 1)
            restricted_mask = in_gene_region & (~self.allowed_gene_dense_mask[clamped_idx])
            if restricted_mask.any():
                r_b, r_l = torch.where(restricted_mask)
                logits[r_b[:, None], r_l[:, None], self.nonzero_count_token_ids[None, :]] = -torch.inf

        return logits

    def _correct_logits_masked_only(
        self,
        logits: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Correct masked-only logits ``(N_mask, D)`` using position info.

        Args:
            logits: ``(N_mask, D)`` — logits only for masked positions.
            masked_positions: ``(N_mask,)`` — original sequence position of
                each logit row.

        Returns:
            Corrected logits (modified in-place).
        """
        in_gene_region = (
            (masked_positions >= self.condition_prefix_length)
            & (masked_positions < self.seq_length)
        )
        gene_indices = torch.where(in_gene_region)[0]

        if "default" in self.strategies and len(gene_indices) > 0:
            row_expand = gene_indices.unsqueeze(1).expand(-1, len(self.non_count_cols))
            col_expand = self.non_count_cols.unsqueeze(0).expand(len(gene_indices), -1)
            logits[row_expand, col_expand] = -torch.inf

        if "restrict_valid_genes" in self.strategies and self.allowed_gene_dense_mask is not None and len(gene_indices) > 0:
            dense_idx = (masked_positions[gene_indices] - self.condition_prefix_length)
            clamped_idx = dense_idx.clamp(min=0, max=self.allowed_gene_dense_mask.shape[0] - 1)
            restricted = ~self.allowed_gene_dense_mask[clamped_idx]
            if restricted.any():
                restricted_indices = gene_indices[restricted]
                row_expand = restricted_indices.unsqueeze(1).expand(-1, len(self.nonzero_count_token_ids))
                col_expand = self.nonzero_count_token_ids.unsqueeze(0).expand(len(restricted_indices), -1)
                logits[row_expand, col_expand] = -torch.inf

        return logits

    def correct_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token correction is not supported for dense tokenizers."""
        raise NotImplementedError(
            "DenseLogitCorrector does not support token correction. "
            "Set correct_tokens=False in sampler parameters."
        )
