from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..submodules.mlp.llama_mlp import GptNeoxMLP, LLaMAMLP
from ..submodules.norm.rmsnorm import FusedRMSNorm, RMSNorm


@dataclass(frozen=True)
class SequenceFormatConfig:
    """
    Workflow-injected sequence layout facts derived from the tokenizer.

    These fields describe the stable token layout consumed by the legacy patch
    model family. The train workflow derives them from tokenizer-owned metadata
    and injects them into model construction as a separate object so sequence
    layout does not need to be duplicated in YAML.

    Compatibility note:
    - `target_gene_high_idx` / `target_gene_low_idx` retain the legacy naming
      because old dense and dense-gene1tok tokenizers shared those field names.
    - For current gene1tok tokenizers, `target_gene_low_idx` is only the
      reserved compatibility placeholder slot.
    - `target_gene_2token_width=None` means the tokenizer family does not use a
      2-token target-gene encoding. It is not an "unknown yet" sentinel.
    """

    condition_prefix_length: int
    seq_length: int
    target_gene_high_idx: int | None
    target_gene_low_idx: int | None
    target_gene_token_id_offset: int | None
    target_gene_2token_width: int | None


@dataclass
class TransEncoderConfig:
    """
    Legacy-style config object for the default patch-dense train model.

    Legacy counterpart:
    - `archive/legacy/lit_gpt/config_transmdm.py:Config`

    Current Stage 3 scope:
    - Keep preset selection and config-level parameter derivation local to the
      model config object.
    - Keep sequence format as an explicit injected structure rather than
      reintroducing tokenizer auto-load behavior.
    - Provide enough config-level helpers that later implementation work does
      not need to redefine preset/derivation policy.
    """

    name: str
    block_size: int = 4096
    vocab_size: int = 64000
    padding_multiple: int = 64
    padded_vocab_size: int | None = None
    n_layer: int = 13
    n_head: int = 10
    n_embd: int = 640
    rotary_percentage: float = 1.0
    parallel_residual: bool = False
    bias: bool = False
    n_query_groups: int | None = None
    shared_attention_norm: bool = False
    norm_eps: float = 1e-5
    intermediate_size: int | None = 2560
    condense_ratio: int = 1
    attn_module: Literal["TransformerEngine", "XFormers", "FlashAttention2", "FlashAttention3", "Auto"] = "Auto"
    pos_emb: Literal["rope"] = "rope"
    _norm_class: str = "FusedRMSNorm"
    _mlp_class: str = "LLaMAMLP"
    patch_size: int = 8
    patch_seed: int = 0
    patch_token_shuffle_per_block: bool = False
    patch_token_shuffle_seed: int = 0
    use_gene_embedding_in_condition: bool = False
    use_gene_embedding_fusion: bool = False
    gene_embedding_file: str = ""
    gene_embedding_dim: int = 5120
    gene_id_name_csv: str | None = None
    # When True, lm_head is only applied to masked token positions (provided
    # via ModelInput.metadata["mask_indices"]). Reduces computation since
    # only masked logits participate in loss/sampling. No effect at inference.
    use_lm_head_masked_only: bool = False
    sequence_format: SequenceFormatConfig | None = None

    def __post_init__(self) -> None:
        """
        Finalize config-level derived values and validate basic invariants.

        Current Stage 3 behavior is intentionally limited to deterministic
        config derivation. It does not auto-load tokenizer metadata, inspect
        files, or recover any runtime/business logic.
        """
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.pos_emb != "rope":
            raise ValueError("Current patch-dense train mainline only supports pos_emb='rope'.")
        self.resolve_padded_vocab_size()
        self.resolve_n_query_groups()
        self.resolve_intermediate_size()

    @property
    def head_size(self) -> int:
        """Return the per-head hidden size implied by `n_embd / n_head`."""
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **overrides: Any) -> "TransEncoderConfig":
        """
        Build one config object from a registered preset plus explicit overrides.

        Args:
            name: Preset name registered in `name_to_config`.
            **overrides: Explicit YAML- or builder-provided field overrides
                applied on top of the preset defaults.

        Returns:
            One mutable `TransEncoderConfig` object containing preset defaults,
            the provided overrides, and config-level derived values resolved by
            `__post_init__`.
        """
        try:
            preset = dict(name_to_config[name])
        except KeyError as exc:
            raise ValueError(f"Unknown TransEncoder preset name: {name}") from exc
        preset.update(overrides)
        preset["name"] = name
        return cls(**preset)

    def apply_sequence_format(self, sequence_format: SequenceFormatConfig) -> "TransEncoderConfig":
        """
        Attach workflow-derived sequence layout metadata to this config object.

        Args:
            sequence_format: Tokenizer-derived sequence layout facts that the
                model needs for later patch/gene-embedding logic.

        Returns:
            `self`, allowing the builder to apply the sequence format and
            immediately pass the config into model construction.
        """
        self.sequence_format = sequence_format
        return self

    def apply_vocab_size(self, vocab_size: int) -> "TransEncoderConfig":
        """
        Attach the tokenizer-reported vocabulary size to this config object.

        Stage 3 allows tokenizer-side placeholder values such as `-1`. In that
        case, the data-flow channel is considered present even if
        `padded_vocab_size` cannot yet be resolved to a meaningful runtime
        value.
        """
        self.vocab_size = vocab_size
        self.padded_vocab_size = None
        self.resolve_padded_vocab_size()
        return self

    def resolve_padded_vocab_size(self) -> int | None:
        """
        Resolve `padded_vocab_size` from `vocab_size` and `padding_multiple`.

        Returns:
            The final padded vocabulary size stored on the config object, or
            `None` when the current Stage 3 tokenizer still reports a
            placeholder `vocab_size`.
        """
        if self.vocab_size < 0:
            return None
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        return self.padded_vocab_size

    def resolve_n_query_groups(self) -> int:
        """
        Resolve grouped-query attention group count from `n_head`.

        Returns:
            The final `n_query_groups` stored on the config object.
        """
        if self.n_query_groups is None:
            self.n_query_groups = self.n_head
        if self.n_head % self.n_query_groups != 0:
            raise ValueError("n_head must be divisible by n_query_groups.")
        return self.n_query_groups

    def resolve_intermediate_size(self) -> int:
        """
        Resolve the transformer MLP intermediate size when not explicitly set.

        Current Stage 3 policy mirrors the legacy config direction:
        non-LLaMA MLPs may derive `4 * n_embd`; LLaMA-style presets must carry
        an explicit intermediate size.
        """
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("LLaMAMLP presets must provide intermediate_size explicitly.")
            self.intermediate_size = 4 * self.n_embd
        return self.intermediate_size

    def resolve_norm_identifier(self) -> str:
        """
        Return the stable norm implementation identifier encoded in config.

        This is a config-level lookup helper only. Stage 3 does not require it
        to import or return the real class object.
        """
        return self._norm_class

    def resolve_mlp_identifier(self) -> str:
        """
        Return the stable MLP implementation identifier encoded in config.

        This is a config-level lookup helper only. Stage 3 does not require it
        to import or return the real class object.
        """
        return self._mlp_class

    @property
    def norm_class(self) -> type[torch.nn.Module]:
        """
        Resolve the concrete norm skeleton class referenced by `_norm_class`.

        This keeps the legacy `config.norm_class(...)` call shape while
        constraining class resolution to the current local `models` module.
        """
        if self._norm_class in _NORM_CLASS_REGISTRY:
            return _NORM_CLASS_REGISTRY[self._norm_class]
        try:
            norm_class = getattr(torch.nn, self._norm_class)
        except AttributeError as exc:
            raise ValueError(f"Unknown norm class identifier: {self._norm_class}") from exc
        if not issubclass(norm_class, torch.nn.Module):
            raise TypeError(f"Resolved norm class is not a torch.nn.Module: {self._norm_class}")
        return norm_class

    @property
    def mlp_class(self) -> type[torch.nn.Module]:
        """
        Resolve the concrete MLP skeleton class referenced by `_mlp_class`.

        Current Stage 3 only supports the local MLP skeletons required by the
        default patch-dense model.
        """
        try:
            return _MLP_CLASS_REGISTRY[self._mlp_class]
        except KeyError as exc:
            raise ValueError(f"Unknown MLP class identifier: {self._mlp_class}") from exc


def find_multiple(value: int, multiple: int) -> int:
    """Return the smallest multiple of `multiple` greater than or equal to `value`."""
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + multiple - remainder
TRANSENCODER_PRESET_CONFIGS: list[dict[str, Any]] = [
    # Preset naming convention: the number (e.g. "11M") refers to transformer
    # block parameters only, excluding embedding, lm_head, patch_proj, and epatch_expand layers.
    dict(
        name="Diff_LLaMA_6M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=6,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        n_query_groups=4,
    ),
    dict(
        name="Diff_LLaMA_11M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=7,
        n_head=5,
        n_embd=320,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1280,
        n_query_groups=5,
    ),
    dict(
        name="Diff_LLaMA_19M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=8,
        n_head=6,
        n_embd=384,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1536,
        n_query_groups=6,
    ),
    dict(
        name="Diff_LLaMA_34M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=8,
        n_head=8,
        n_embd=512,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        n_query_groups=8,
    ),
    dict(
        name="Diff_LLaMA_48M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=9,
        n_head=9,
        n_embd=576,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2304,
        n_query_groups=9,
    ),
    dict(
        name="Diff_LLaMA_66M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=10,
        n_head=10,
        n_embd=640,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2560,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_75M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=16,
        n_head=8,
        n_embd=640,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1600,
        n_query_groups=8,
    ),
    dict(
        name="Diff_LLaMA_85M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=13,
        n_head=10,
        n_embd=640,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2560,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_113M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3072,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_142M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=15,
        n_head=12,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3072,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_170M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=18,
        n_head=12,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3072,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_180M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=14,
        n_head=14,
        n_embd=896,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3584,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_206M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=16,
        n_head=14,
        n_embd=896,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3584,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_231M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=18,
        n_head=14,
        n_embd=896,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3584,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_268M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=16,
        n_head=16,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_302M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=18,
        n_head=16,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_336M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=20,
        n_head=16,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_472M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=18,
        n_head=10,
        n_embd=1280,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5120,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_551M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=21,
        n_head=10,
        n_embd=1280,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5120,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_571M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=18,
        n_head=11,
        n_embd=1408,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=11,
    ),
    dict(
        name="Diff_LLaMA_629M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=24,
        n_head=10,
        n_embd=1280,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5120,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_666M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=21,
        n_head=11,
        n_embd=1408,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=11,
    ),
    dict(
        name="Diff_LLaMA_717M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=19,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_761M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=24,
        n_head=11,
        n_embd=1408,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=11,
    ),
    dict(
        name="Diff_LLaMA_831M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=22,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_944M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=25,
        n_head=12,
        n_embd=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_1028M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=20,
        n_head=14,
        n_embd=1792,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=7168,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_1233M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=24,
        n_head=14,
        n_embd=1792,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=7168,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_1476M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=22,
        n_head=16,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=8192,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_1678M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=25,
        n_head=16,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=8192,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_2121M",
        block_size=4096,
        vocab_size=64000,
        padding_multiple=64,
        n_layer=28,
        n_head=17,
        n_embd=2176,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=8704,
        n_query_groups=17,
    ),
]

name_to_config: dict[str, dict[str, Any]] = {
    config["name"]: config for config in TRANSENCODER_PRESET_CONFIGS
}

_NORM_CLASS_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "RMSNorm": RMSNorm,
    "FusedRMSNorm": FusedRMSNorm,
}

_MLP_CLASS_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "GptNeoxMLP": GptNeoxMLP,
    "LLaMAMLP": LLaMAMLP,
}
