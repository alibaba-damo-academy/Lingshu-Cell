"""Standalone helper functions for the reverse masked-diffusion sampler.

Extracted from ``DefaultSampler`` so they can be called directly by the
inference workflow without a wrapper class.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional

import torch

from ...common.interfaces import ModelInput, ModelOutput
from .logit_token_corrector import DenseLogitCorrector, SparseLogitCorrector

# Registry mapping config string names to corrector classes.
# The ``Batch*`` / ``LogitAndTokenCorrector`` aliases are kept for
# backward-compatibility with older config files (deprecated).
CORRECTOR_REGISTRY: Mapping[str, type] = {
    "DenseLogitCorrector": DenseLogitCorrector,
    "BatchDenseLogitAndTokenCorrector": DenseLogitCorrector,  # deprecated alias
    "SparseLogitCorrector": SparseLogitCorrector,
    "LogitAndTokenCorrector": SparseLogitCorrector,  # deprecated alias
}


def forward_model(
    model: Any,
    model_input: ModelInput,
    *,
    device_type: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Run a single model forward pass wrapped in ``torch.amp.autocast``.

    The caller is responsible for constructing a ``ModelInput`` containing
    all necessary metadata (``mask_indices``, ``drug_indices``,
    ``dose_values``, etc.) before calling this function.

    Args:
        model: Model accepting ``ModelInput`` and returning ``ModelOutput``.
        model_input: Fully assembled ``ModelInput`` with
            ``noisy_token_ids`` and ``metadata``.
        device_type: Autocast device type (``"cuda"``, ``"cpu"``, etc.).
        dtype: Autocast dtype (e.g. ``torch.bfloat16``).

    Returns:
        Tuple of ``(logits, is_logits_masked_only)``.
        - When ``is_logits_masked_only=False``: logits is ``(B, L, V)``.
        - When ``is_logits_masked_only=True``: logits is ``(N_mask, V)``.
    """
    with torch.amp.autocast(device_type, dtype=dtype):
        output: ModelOutput = model(model_input)
    is_masked_only = output.metadata.get("is_logits_masked_only", False)
    return output.logits, is_masked_only


def forward_with_cfg(
    model: Any,
    model_input: ModelInput,
    cfg_uncond_input_generator: Callable[[ModelInput], ModelInput],
    cfg_scale: float,
    *,
    device_type: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Run model forward pass with classifier-free guidance.

    Constructs unconditional input via ``cfg_uncond_input_generator``,
    concatenates both into a single ``ModelInput`` using
    ``ModelInput.cat`` (which batch-aligns all metadata tensor fields),
    runs one forward pass, then applies the CFG formula:
    ``logits = un_logits + (cfg_scale + 1) * (cond_logits - un_logits)``

    Args:
        model: Model accepting ``ModelInput`` and returning ``ModelOutput``.
        model_input: Conditional ``ModelInput`` (``noisy_token_ids`` +
            all metadata including ``mask_indices`` if applicable).
        cfg_uncond_input_generator: Callable that transforms a conditional
            ``ModelInput`` into an unconditional one (e.g. replaces
            ``drug_indices`` with DMSO control index, swaps condition
            prefix tokens, etc.). Must preserve non-condition metadata
            like ``mask_indices``.
        cfg_scale: Guidance scale.
        device_type: Autocast device type.
        dtype: Autocast dtype.

    Returns:
        Tuple of ``(guided_logits, is_logits_masked_only)``.
    """
    uncond_input = cfg_uncond_input_generator(model_input)
    combined = ModelInput.cat([model_input, uncond_input], dim=0)

    combined_logits, is_masked_only = forward_model(
        model, combined, device_type=device_type, dtype=dtype,
    )

    b = len(model_input)
    if is_masked_only:
        assert combined_logits.shape[0] % 2 == 0, (
            "Masked-only CFG logits must have even count "
            "(cond+uncond must have same mask)."
        )
        half = combined_logits.shape[0] // 2
        cond_logits = combined_logits[:half]
        un_logits = combined_logits[half:]
    else:
        cond_logits = combined_logits[:b]
        un_logits = combined_logits[b:]

    return un_logits + (cfg_scale + 1) * (cond_logits - un_logits), is_masked_only


def build_corrector(
    corrector_cls_name: str,
    *,
    seq_shape: tuple[int, int],
    logit_dim: int,
    tokenizer: Any,
    strategies: str,
    device: str,
    sample_gene_names: Optional[List[str]] = None,
) -> Optional[object]:
    """Instantiate a logit/token corrector from the registry.

    Args:
        corrector_cls_name: Key in ``CORRECTOR_REGISTRY``.
        seq_shape: ``(B, L)`` sequence shape.
        logit_dim: Vocabulary size (logit dimension).
        tokenizer: Tokenizer instance forwarded to the corrector.
        strategies: Comma-separated strategy string (e.g. ``"default"``).
        device: Torch device string.
        sample_gene_names: Optional gene names for ``restrict_valid_genes``.

    Returns:
        Corrector instance, or ``None`` if ``strategies`` is empty.
    """
    if not strategies or strategies == "":
        return None

    corrector_cls = CORRECTOR_REGISTRY.get(corrector_cls_name)
    if corrector_cls is None:
        raise ValueError(
            f"Unknown corrector class: {corrector_cls_name!r}. "
            f"Available: {list(CORRECTOR_REGISTRY.keys())}"
        )

    kwargs: dict[str, Any] = {
        "seq_shape": seq_shape,
        "logit_dim": logit_dim,
        "tokenizer": tokenizer,
        "strategies": strategies,
        "device": device,
    }

    if sample_gene_names is not None:
        kwargs["sample_gene_names"] = sample_gene_names

    return corrector_cls(**kwargs)
