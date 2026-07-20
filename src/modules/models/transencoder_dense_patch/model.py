from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from ...common.interfaces import ModelInput, ModelOutput
from ..submodules.rotary_emb.fused_rotary_embedding import RoPECache, build_rope_cache
from .block import Block
from .config import TransEncoderConfig


class TransEncoder(torch.nn.Module):
    """
    Patch-dense train model aligned to the legacy VCC mainline.

    Explicit non-goals:
    - `only_attend_to_condition`
    """

    def __init__(self, config: TransEncoderConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        if config.pos_emb != "rope":
            raise ValueError(f"Unsupported position_encoding for patch model: {config.pos_emb}")

        self.config = config
        self.sequence_format = config.sequence_format
        self.vocab_size = config.vocab_size
        self.padded_vocab_size = config.padded_vocab_size
        self.patch_size = config.patch_size
        self.patch_seed = config.patch_seed
        if self.patch_size <= 1:
            raise ValueError(f"{self.__class__.__name__} requires config.patch_size > 1, got {self.patch_size}")

        if config.sequence_format is None:
            raise ValueError("TransEncoder requires workflow-injected SequenceFormatConfig.")
        self.condition_prefix_length = int(config.sequence_format.condition_prefix_length)
        self.seq_length = int(config.sequence_format.seq_length)
        if self.seq_length <= self.condition_prefix_length:
            raise ValueError("Sequence format gene segment must be positive.")

        self.vocab_n_embed = config.n_embd
        self.attn_module = config.attn_module
        self.position_embedding = config.pos_emb
        self.lm_head = torch.nn.Linear(self.vocab_n_embed, config.padded_vocab_size, bias=False)
        self.patch_proj = torch.nn.Linear(self.patch_size * self.vocab_n_embed, self.vocab_n_embed, bias=False)
        self.depatch_expand = torch.nn.Linear(self.vocab_n_embed, self.patch_size * self.vocab_n_embed, bias=False)
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.padded_vocab_size + 1, self.vocab_n_embed),
                h=torch.nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None

        self.gene_len = self.seq_length - self.condition_prefix_length
        self.gene_pad_len = (-self.gene_len) % self.patch_size
        self.gene_len_padded = self.gene_len + self.gene_pad_len
        self.num_patches = self.gene_len_padded // self.patch_size
        if int(config.block_size) < self.seq_length:
            raise ValueError(f"config.block_size ({config.block_size}) must be >= seq_length ({self.seq_length})")
        tail_pad_len = int(config.block_size - self.seq_length)
        if tail_pad_len < self.gene_pad_len:
            raise ValueError(
                f"Not enough tail padding tokens for patch padding: need {self.gene_pad_len}, got {tail_pad_len}."
            )
        self.patch_block_size = (
            int(self.condition_prefix_length)
            + int(self.num_patches)
            + int(tail_pad_len - self.gene_pad_len)
        )

        # Fixed random binning (permutation) for gene tokens
        g = torch.Generator()
        g.manual_seed(self.patch_seed)
        gene_perm = torch.randperm(self.gene_len, generator=g, device=g.device)
        gene_inv_perm = torch.empty_like(gene_perm)
        gene_inv_perm[gene_perm] = torch.arange(self.gene_len, device=gene_perm.device)
        self.register_buffer("gene_perm", gene_perm, persistent=True)
        self.register_buffer("gene_inv_perm", gene_inv_perm, persistent=True)

        self.patch_token_shuffle_per_block = config.patch_token_shuffle_per_block
        self.patch_token_shuffle_seed = config.patch_token_shuffle_seed
        if self.patch_token_shuffle_per_block:
            g_patch = torch.Generator()
            g_patch.manual_seed(self.patch_token_shuffle_seed)
            g_device = g_patch.device
            patch_perm_by_layer = torch.empty(
                (int(config.n_layer), int(self.num_patches)), dtype=torch.long, device=g_device
            )
            patch_inv_perm_by_layer = torch.empty_like(patch_perm_by_layer)
            for layer_idx in range(int(config.n_layer)):
                p = torch.randperm(int(self.num_patches), generator=g_patch, device=g_device)
                inv_p = torch.empty_like(p)
                inv_p[p] = torch.arange(int(self.num_patches), dtype=torch.long, device=g_device)
                patch_perm_by_layer[layer_idx] = p
                patch_inv_perm_by_layer[layer_idx] = inv_p
            self.register_buffer("patch_perm_by_layer", patch_perm_by_layer, persistent=True)
            self.register_buffer("patch_inv_perm_by_layer", patch_inv_perm_by_layer, persistent=True)

        self.use_gene_embedding_in_condition = bool(config.use_gene_embedding_in_condition)
        self.use_gene_embedding_fusion = bool(config.use_gene_embedding_fusion)
        if self.use_gene_embedding_in_condition or self.use_gene_embedding_fusion:
            self._init_gene_embeddings()
            self.gene_embedding_projection = torch.nn.Linear(config.gene_embedding_dim, config.n_embd // 2, bias=False)
            self.target_gene_high_idx = config.sequence_format.target_gene_high_idx
            self.target_gene_low_idx = config.sequence_format.target_gene_low_idx
            # Legacy patch models write the replacement embedding to the last
            # prefix slot. Keep that convention for the current VCC mainline.
            self.target_gene_emb_idx = self.condition_prefix_length - 1
            self.target_gene_token_id_offset = config.sequence_format.target_gene_token_id_offset
            self.target_gene_2token_width = config.sequence_format.target_gene_2token_width

        self.apply(lambda module: self._init_weights(module, int(config.n_layer)))

    def _init_gene_embeddings(self) -> None:
        config = self.config
        if not (config.gene_embedding_file and config.gene_id_name_csv):
            raise ValueError(
                "Gene embedding features require both gene_embedding_file and gene_id_name_csv."
            )

        target_mean = 0.0
        target_std = math.sqrt(2.0 / 5 / config.n_embd)

        gene_mapping_df = pd.read_csv(
            config.gene_id_name_csv,
            sep="\t",
            header=None,
            names=["ensembl_id", "gene_name"],
        )
        ensembl_id_to_gene_name = dict(zip(gene_mapping_df["ensembl_id"], gene_mapping_df["gene_name"]))
        gene_names_ordered = gene_mapping_df["gene_name"].tolist()

        with np.load(config.gene_embedding_file) as data:
            gene_embeddings_data = {
                ensembl_id_to_gene_name[ensembl_id]: torch.from_numpy(data[ensembl_id].astype(np.float32))
                for ensembl_id in data.files
                if ensembl_id in ensembl_id_to_gene_name
            }

        if not gene_embeddings_data:
            raise ValueError(f"No gene embeddings could be loaded from {config.gene_embedding_file}.")

        loaded_embeddings = torch.stack(list(gene_embeddings_data.values()))
        loaded_mean_per_dim = loaded_embeddings.mean(dim=0)
        loaded_std_per_dim = loaded_embeddings.std(dim=0)

        for gene_name, embedding in list(gene_embeddings_data.items()):
            gene_embeddings_data[gene_name] = (
                (embedding - loaded_mean_per_dim) / (loaded_std_per_dim + 1e-5) * target_std + target_mean
            )

        gene_embeddings_list = []
        for gene_name in gene_names_ordered:
            if gene_name in gene_embeddings_data:
                gene_embedding = gene_embeddings_data[gene_name]
            else:
                seed = int(hashlib.md5(gene_name.encode()).hexdigest()[:8], 16)
                generator = torch.Generator().manual_seed(seed)
                gene_embedding = torch.normal(
                    mean=target_mean,
                    std=target_std,
                    size=(config.gene_embedding_dim,),
                    generator=generator,
                    dtype=torch.float32,
                )
            gene_embeddings_list.append(gene_embedding)

        self.register_buffer("gene_embeddings", torch.stack(gene_embeddings_list), persistent=True)

    def _init_weights(self, module: torch.nn.Module, n_layer: int) -> None:
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, torch.nn.Linear):
            if hasattr(self, "gene_embedding_projection") and module is self.gene_embedding_projection:
                fan_in = self.config.gene_embedding_dim
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if module.__class__.__name__ == "SelfAttention" and hasattr(module, "proj"):
            torch.nn.init.normal_(
                module.proj.weight,
                mean=0.0,
                std=1 / math.sqrt(self.config.n_embd) / n_layer,
            )

    def build_rope_cache(self, device: torch.device, dtype: torch.dtype) -> RoPECache:
        return build_rope_cache(
            seq_len=self.patch_block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=dtype,
            device=device,
            condense_ratio=self.config.condense_ratio,
        )

    def _apply_gene_embedding_in_condition(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "gene_embeddings"):
            return x
        if self.target_gene_high_idx is None:
            return x

        if self.target_gene_2token_width is not None:
            if self.target_gene_low_idx is None:
                raise ValueError(
                    "target_gene_low_idx must be set when target_gene_2token_width is used."
                )
            gene_high = idx[:, self.target_gene_high_idx] - int(self.target_gene_token_id_offset or 0)
            gene_low = idx[:, self.target_gene_low_idx] - int(self.target_gene_token_id_offset or 0)
            valid_gene_mask = (gene_high >= 0) & (gene_high < int(self.target_gene_2token_width))
            if valid_gene_mask.any():
                gene_indices = (
                    int(self.target_gene_2token_width) * gene_high + gene_low
                )[valid_gene_mask]
                gene_embs = self.gene_embeddings[gene_indices]
                projected_gene_emb = self.gene_embedding_projection(gene_embs)
                repeated_gene_emb = projected_gene_emb.repeat(1, 2)
                x[valid_gene_mask, self.target_gene_emb_idx] = repeated_gene_emb.to(x)
            return x

        gene_idx = idx[:, self.target_gene_high_idx] - int(self.target_gene_token_id_offset or 0)
        valid_gene_mask = (gene_idx >= 0) & (gene_idx < self.gene_embeddings.shape[0])
        if valid_gene_mask.any():
            gene_indices = gene_idx[valid_gene_mask]
            gene_embs = self.gene_embeddings[gene_indices]
            projected_gene_emb = self.gene_embedding_projection(gene_embs)
            repeated_gene_emb = projected_gene_emb.repeat(1, 2)
            x[valid_gene_mask, self.target_gene_emb_idx] = repeated_gene_emb.to(x)
        return x

    def _apply_gene_embedding_fusion(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "gene_embeddings"):
            return x
        count_start_pos = self.condition_prefix_length
        count_end_pos = self.seq_length
        count_embs_half = x[:, count_start_pos:count_end_pos, : self.config.n_embd // 2]
        projected_gene_embs = self.gene_embedding_projection(self.gene_embeddings)
        fused_emb = torch.cat(
            [count_embs_half, projected_gene_embs.unsqueeze(0).expand(x.size(0), -1, -1)],
            dim=-1,
        )
        x[:, count_start_pos:count_end_pos] = fused_emb.to(x)
        return x

    def forward(self, inputs: ModelInput) -> ModelOutput:
        token_ids = torch.as_tensor(inputs.noisy_token_ids, dtype=torch.long, device=self.lm_head.weight.device)
        batch_size, total_length = token_ids.shape
        if total_length < self.seq_length:
            raise ValueError(f"TransEncoder input length must be >= {self.seq_length}, got {total_length}.")

        x = self.transformer.wte(token_ids)
        if self.use_gene_embedding_in_condition:
            x = self._apply_gene_embedding_in_condition(x, token_ids)
        if self.use_gene_embedding_fusion:
            x = self._apply_gene_embedding_fusion(x)

        cond_x = x[:, : self.condition_prefix_length, :]
        gene_x = x[:, self.condition_prefix_length : self.seq_length, :]
        tail_pad_x = x[:, self.seq_length :, :]

        if tail_pad_x.shape[1] < self.gene_pad_len:
            raise ValueError(
                f"Not enough tail padding tokens for patch padding: need {self.gene_pad_len}, got {tail_pad_x.shape[1]}."
            )

        gene_x = gene_x.index_select(dim=1, index=self.gene_perm)
        if self.gene_pad_len > 0:
            gene_x = torch.cat([gene_x, tail_pad_x[:, : self.gene_pad_len, :]], dim=1)
            tail_pad_x = tail_pad_x[:, self.gene_pad_len :, :]

        gene_x = gene_x.reshape(batch_size, self.num_patches, self.patch_size * self.vocab_n_embed)
        patch_x = self.patch_proj(gene_x)
        x_patch = torch.cat([cond_x, patch_x, tail_pad_x], dim=1)

        rope = None
        if self.rope_cache is None or self.rope_cache[0].device != x_patch.device:
            self.rope_cache = self.build_rope_cache(device=x_patch.device, dtype=torch.bfloat16)
        cos, sin = self.rope_cache
        rope = (cos[: x_patch.size(1)], sin[: x_patch.size(1)])

        if self.patch_token_shuffle_per_block:
            prefix = self.condition_prefix_length
            num_patches = int(self.num_patches)
            for layer_idx, block in enumerate(self.transformer.h):
                perm = self.patch_perm_by_layer[layer_idx]
                inv_perm = self.patch_inv_perm_by_layer[layer_idx]
                cond_in = x_patch[:, :prefix, :]
                patch_in = x_patch[:, prefix : prefix + num_patches, :].index_select(dim=1, index=perm)
                tail_in = x_patch[:, prefix + num_patches :, :]
                x_in = torch.cat([cond_in, patch_in, tail_in], dim=1)
                x_out = block(x_in, rope)
                cond_out = x_out[:, :prefix, :]
                patch_out = x_out[:, prefix : prefix + num_patches, :].index_select(dim=1, index=inv_perm)
                tail_out = x_out[:, prefix + num_patches :, :]
                x_patch = torch.cat([cond_out, patch_out, tail_out], dim=1)
        else:
            for block in self.transformer.h:
                x_patch = block(x_patch, rope)

        x_patch = self.transformer.ln_f(x_patch)
        cond_out = x_patch[:, : self.condition_prefix_length, :]
        patch_out = x_patch[:, self.condition_prefix_length : self.condition_prefix_length + self.num_patches, :]
        tail_pad_out = x_patch[:, self.condition_prefix_length + self.num_patches :, :]

        patch_out = self.depatch_expand(patch_out)
        gene_out = patch_out.reshape(batch_size, self.num_patches, self.patch_size, self.vocab_n_embed)
        gene_out = gene_out.reshape(batch_size, self.gene_len_padded, self.vocab_n_embed)
        gene_out_real = gene_out[:, : self.gene_len, :].index_select(dim=1, index=self.gene_inv_perm)
        gene_out_pad = gene_out[:, self.gene_len :, :]
        x_out = torch.cat([cond_out, gene_out_real, gene_out_pad, tail_pad_out], dim=1)
        if x_out.shape[1] != total_length:
            raise RuntimeError(f"Unexpected output length: got {x_out.shape[1]}, expected {total_length}.")
        mask_indices = inputs.metadata.get("mask_indices")
        if self.config.use_lm_head_masked_only and mask_indices is not None:
            logits = self.lm_head(x_out[mask_indices])
            meta = dict(inputs.metadata)
            meta["is_logits_masked_only"] = True
            return ModelOutput(logits=logits, metadata=meta)
        else:
            logits = self.lm_head(x_out)
            meta = dict(inputs.metadata)
            meta["is_logits_masked_only"] = False
            return ModelOutput(logits=logits, metadata=meta)
