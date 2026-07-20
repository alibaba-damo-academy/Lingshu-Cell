"""Sparse logit and token corrector for batch-aware reverse diffusion sampling.

Operates on full-sequence logits of shape ``(B, L, D)`` for sparse
tokenizers that use alternating gene/count positions (even = gene, odd =
count).

Ported from the authoritative ``LogitAndTokenCorrector`` in
``TransMDM_VCC/eval/batch_default_sampler.py`` (L167-375).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class SparseLogitCorrector:
    """Batch-compatible logit and token corrector for sparse (gene+count pairs) tokenizers.

    Strategies:
        ``default``:
            Even positions accept only gene tokens + EOS; odd positions
            accept only count tokens + EOS.
        ``unique_gene``:
            Prevents sampling duplicate gene tokens within a sample.
            For tokens: masks repeated gene tokens after the first occurrence.
        ``continuous_eos``:
            Once an EOS token appears, forces all subsequent masked
            positions to sample EOS. For tokens: sets all unmasked
            tokens after the first EOS to EOS.
        ``min_length``:
            Prevents EOS from being sampled at positions before
            ``min_length``.

    Args:
        seq_shape: ``(B, L)`` — batch size and padded sequence length.
        logit_dim: Vocabulary size (``D``), excluding the mask token.
        tokenizer: Sparse tokenizer instance. Must expose
            ``min_gene_token_id``, ``max_gene_token_id``,
            ``min_count_token_id``, ``max_count_token_id``, ``eos_id``.
        strategies: Comma-separated strategy string.
        ref_gene_token_ids: Optional allow-list of valid gene token ids.
        min_length: Minimum sequence length for the ``min_length`` strategy.
        device: Torch device.
    """

    def __init__(
        self,
        seq_shape: Tuple[int, int],
        logit_dim: int,
        tokenizer: object,
        strategies: str = "default",
        ref_gene_token_ids: Optional[List[int]] = None,
        min_length: int = 0,
        device: str = "cuda",
    ) -> None:
        parsed = strategies.split(",")
        for s in parsed:
            assert s in ("", "default", "unique_gene", "continuous_eos", "min_length"), (
                f"Unknown strategy: {s!r}"
            )
        if "min_length" in parsed:
            assert min_length is not None and min_length >= 0, "min_length is not correctly set."
        self.strategies = parsed
        if self.strategies == [""]:
            return

        self.B = seq_shape[0]
        self.L = seq_shape[1]
        self.D = logit_dim
        self.min_length = min_length
        self.mask_id = logit_dim
        self.device = device

        self.min_gene_token_id = tokenizer.min_gene_token_id
        self.max_gene_token_id = tokenizer.max_gene_token_id
        self.min_count_token_id = tokenizer.min_count_token_id
        self.max_count_token_id = tokenizer.max_count_token_id
        self.eos_id = tokenizer.eos_id

        logit_indices = torch.arange(logit_dim, device=device)
        self.positions = (
            torch.arange(self.L, dtype=torch.long, device=device)
            .reshape(1, -1)
            .repeat(self.B, 1)
        )

        if ref_gene_token_ids is not None:
            ref_gene_logit_mask = torch.zeros(logit_dim, dtype=torch.bool, device=device)
            ref_gene_logit_mask[ref_gene_token_ids] = True
            self.gene_mask = ~(
                (
                    (logit_indices >= self.min_gene_token_id)
                    & (logit_indices <= self.max_gene_token_id)
                    & ref_gene_logit_mask
                )
                | (logit_indices == self.eos_id)
            )
        else:
            self.gene_mask = ~(
                (
                    (logit_indices >= self.min_gene_token_id)
                    & (logit_indices <= self.max_gene_token_id)
                )
                | (logit_indices == self.eos_id)
            )

        self.count_mask = ~(
            (
                (logit_indices >= self.min_count_token_id)
                & (logit_indices <= self.max_count_token_id)
            )
            | (logit_indices == self.eos_id)
        )

    def correct_logits(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Correct logits in-place for a batch of sequences.

        Args:
            logits: ``(B, L, D)`` — full-position logits.
            tokens: ``(B, L)`` — current token ids (masked and unmasked).

        Returns:
            The corrected ``logits`` tensor (modified in-place).
        """
        if "default" in self.strategies:
            is_gene_pos = self.positions % 2 == 0
            is_count_pos = self.positions % 2 == 1

            gene_correction_mask = is_gene_pos.unsqueeze(-1) & self.gene_mask.view(1, 1, -1)
            count_correction_mask = is_count_pos.unsqueeze(-1) & self.count_mask.view(1, 1, -1)
            logits[gene_correction_mask | count_correction_mask] = -torch.inf

        if "unique_gene" in self.strategies:
            is_gene_mask = (tokens >= self.min_gene_token_id) & (tokens <= self.max_gene_token_id)
            is_unmasked_mask = tokens != self.mask_id
            combined_mask = is_gene_mask & is_unmasked_mask

            for i in range(self.B):
                existing_genes_in_sample = torch.unique(tokens[i, combined_mask[i]])
                if existing_genes_in_sample.numel() > 0:
                    logits[i, :, existing_genes_in_sample] = -torch.inf

        if "continuous_eos" in self.strategies:
            is_eos = tokens == self.eos_id
            if is_eos.any():
                mask_at_or_after_eos = torch.cumsum(is_eos, dim=1) > 0
                mask_after_eos = torch.zeros_like(mask_at_or_after_eos)
                mask_after_eos[:, 1:] = mask_at_or_after_eos[:, :-1]

                is_masked = tokens == self.mask_id
                correction_positions = mask_after_eos & is_masked

                if torch.any(correction_positions):
                    cb, cl = torch.where(correction_positions)
                    logits[cb, cl, :] = -torch.inf
                    logits[cb, cl, self.eos_id] = 0.0

        if "min_length" in self.strategies:
            early_positions_mask = self.positions < self.min_length
            if torch.any(early_positions_mask):
                eb, el = torch.where(early_positions_mask)
                logits[eb, el, self.eos_id] = -torch.inf

        return logits

    def correct_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Correct tokens in-place for a batch of sequences.

        Args:
            tokens: ``(B, L)`` — current token ids (masked and unmasked).

        Returns:
            The corrected ``tokens`` tensor (modified in-place).
        """
        if "unique_gene" in self.strategies:
            for i in range(self.B):
                seq = tokens[i]
                is_gene_mask = (seq >= self.min_gene_token_id) & (seq <= self.max_gene_token_id)
                is_unmasked_mask = seq != self.mask_id
                combined_mask = is_gene_mask & is_unmasked_mask

                if not torch.any(combined_mask):
                    continue

                gene_tokens_in_seq = seq[combined_mask]
                original_indices_of_genes = torch.where(combined_mask)[0]

                sorted_genes, sort_indices = torch.sort(gene_tokens_in_seq, stable=True)

                is_first_occurrence_sorted = torch.cat(
                    (
                        torch.tensor([True], device=tokens.device),
                        sorted_genes[1:] != sorted_genes[:-1],
                    )
                )

                indices_of_duplicate_in_gene_subseq = sort_indices[~is_first_occurrence_sorted]
                indices_to_mask_in_original_seq = original_indices_of_genes[
                    indices_of_duplicate_in_gene_subseq
                ]

                if indices_to_mask_in_original_seq.numel() > 0:
                    tokens[i, indices_to_mask_in_original_seq] = self.mask_id

        if "continuous_eos" in self.strategies:
            is_eos = tokens == self.eos_id

            mask_at_or_after_eos = torch.cumsum(is_eos, dim=1) > 0
            mask_after_eos = torch.zeros_like(mask_at_or_after_eos)
            mask_after_eos[:, 1:] = mask_at_or_after_eos[:, :-1]

            is_unmasked = tokens != self.mask_id
            final_correction_mask = mask_after_eos & is_unmasked
            tokens[final_correction_mask] = self.eos_id

        return tokens
