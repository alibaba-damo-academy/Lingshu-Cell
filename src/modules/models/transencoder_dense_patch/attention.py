from __future__ import annotations

import importlib.util
import math
from typing import Any, Optional, Tuple

import torch

from .config import TransEncoderConfig
from ..submodules.rotary_emb.fused_rotary_embedding import (
    RoPECache, 
    apply_rotary_emb_func
)

try:  # pragma: no cover - optional dependency
    import xformers.ops
except Exception:  # pragma: no cover - optional dependency
    xformers = None

try:  # pragma: no cover - optional dependency
    import transformer_engine.pytorch as te
except Exception:  # pragma: no cover - optional dependency
    te = None

_FLASH_ATTENTION_2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
_FLASH_ATTENTION_3_AVAILABLE = importlib.util.find_spec("flash_attn_interface") is not None
_XFORMERS_AVAILABLE = xformers is not None
_TE_AVAILABLE = te is not None

class SelfAttention(torch.nn.Module):
    def __init__(self, config: TransEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_query_groups = config.n_query_groups
        self.head_size = config.head_size
        self.bias = config.bias
        self.attn_module = config.attn_module
        self.qkv_projection_shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.qkv = torch.nn.Linear(config.n_embd, self.qkv_projection_shape, bias=config.bias)
        self.proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.te_dot_product_attn = None
        if _TE_AVAILABLE and self.attn_module in ["Auto", "TransformerEngine"]:
            self.te_dot_product_attn = te.DotProductAttention(
                num_attention_heads=config.n_head,
                kv_channels=config.head_size,
                attn_mask_type="no_mask",
                qkv_format="bshd",
            )

    def forward(self, x: torch.Tensor, rope: Optional[RoPECache] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.n_query_groups, total_qkv, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(batch_size, seq_len, self.n_head, self.head_size)
        k = k.reshape(batch_size, seq_len, self.n_query_groups, self.head_size)
        v = v.reshape(batch_size, seq_len, self.n_query_groups, self.head_size)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        y = self.scaled_dot_product_attention(q, k, v)
        y = y.reshape(batch_size, seq_len, hidden_size)
        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_size)
        if self.te_dot_product_attn is not None:
            return self.te_dot_product_attn(q, k, v)
        if _XFORMERS_AVAILABLE and self.attn_module in ["Auto", "XFormers"]:
            if q.size(2) != k.size(2):
                repeat_factor = q.size(2) // k.size(2)
                k = k.repeat_interleave(repeat_factor, dim=2)
                v = v.repeat_interleave(repeat_factor, dim=2)
            return xformers.ops.memory_efficient_attention(q, k, v)
        if _FLASH_ATTENTION_3_AVAILABLE and self.attn_module in ["Auto", "FlashAttention3"]:
            from flash_attn_interface import flash_attn_func

            return flash_attn_func(q, k, v, softmax_scale=scale, causal=False)[0]
        if _FLASH_ATTENTION_2_AVAILABLE and self.attn_module in ["Auto", "FlashAttention2"]:
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size(1) != k.size(1):
            repeat_factor = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            scale=scale,
            is_causal=False,
        )
        return y.transpose(1, 2)
