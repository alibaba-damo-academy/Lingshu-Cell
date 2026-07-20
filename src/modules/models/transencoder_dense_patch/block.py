from __future__ import annotations

from typing import Any

import torch

from .attention import SelfAttention
from .config import TransEncoderConfig


class Block(torch.nn.Module):
    """
    Interface-first transformer block skeleton for the patch-dense model.

    Legacy counterpart:
    - `archive/legacy/lit_gpt/diffmodel_patch.py:Block`

    Current Stage 3 scope:
    - Preserve the legacy `__init__(config)` entrypoint.
    - Record how one block assembles norm -> attention -> optional second norm
      -> MLP under the current patch-dense config object.
    - Keep the residual-path control flags explicit without implementing the
      numerical forward.
    """

    def __init__(self, config: TransEncoderConfig) -> None:
        """
        Initialize one patch-dense transformer block skeleton.

        Args:
            config: Patch-dense model config providing the norm/attention/MLP
                implementation selectors and residual-layout flags.
        """
        super().__init__()
        self.config = config
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        self.norm_2 = None
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(
            hidden_size=config.n_embd,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )
        self.parallel_residual = config.parallel_residual
        self.shared_attention_norm = config.shared_attention_norm

    def forward(self, x: torch.Tensor, rope: Any | None = None) -> torch.Tensor:
        """
        Run one transformer block over hidden states.

        Args:
            x: Hidden states aligned with the token axis, typically shaped
                `B x L x D`.
            rope: Optional RoPE cache or equivalent positional-rotation object
                forwarded to the attention module.

        Returns:
            Hidden states with the same semantic shape as `x`, still aligned as
            `B x L x D`, after one attention + MLP residual block.
        """
        normed = self.norm_1(x)
        attn_out = self.attn(normed, rope)
        if self.parallel_residual:
            mlp_input = normed if self.shared_attention_norm else self.norm_2(x)
            mlp_out = self.mlp(mlp_input)
            return x + attn_out + mlp_out

        if self.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses non-parallel residual and shared attention norm."
            )
        x = x + attn_out
        mlp_input = self.norm_2(x)
        mlp_out = self.mlp(mlp_input)
        return x + mlp_out
