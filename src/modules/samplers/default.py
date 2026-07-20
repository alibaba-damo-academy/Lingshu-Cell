"""Core reverse masked-diffusion sampling function.

Implements the iterative unmasking loop for inference. Each step:
1. Identifies remaining mask positions.
2. Runs the model (optionally with classifier-free guidance).
3. Applies logit correction.
4. Selects which masks to unmask and samples replacement tokens.

Ported and extended from the authoritative batch sampler at
``TransMDM_VCC/eval/batch_default_sampler.py``, with the following changes:

- Adapted to use ``ModelInput`` / ``ModelOutput`` interfaces.
- Added batch-mode CFG support (the legacy batch sampler had
  ``raise NotImplementedError``).
- Autocast wrapping with caller-supplied ``device_type`` / ``dtype``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from tqdm import tqdm

from ..common.interfaces import ModelInput
from .utils.sampling_helpers import build_corrector, forward_model, forward_with_cfg
from .utils.timestep_schedule import (
    categorical_sample_by_gumbel_max,
    get_timesteps,
    get_unmask_token_numbers,
)


@torch.no_grad()
def diff_sample(
    *,
    model: Any,
    initial_input: ModelInput,
    mask_token_id: int,
    config: Any,
    tokenizer: Any,
    device: str,
    device_type: str,
    dtype: torch.dtype,
    cfg_uncond_input_generator: Optional[Callable[[ModelInput], ModelInput]] = None,
    corrector_sample_gene_names: Optional[list[str]] = None,
    temperature: float = 1.0,
    print_info: bool = False,
) -> torch.Tensor:
    """Core reverse diffusion sampling loop.

    All schedule and corrector parameters are read directly from ``config``
    (an OmegaConf namespace or similar). Runtime-only parameters that do not
    belong in config are passed as explicit keyword arguments.

    Args:
        model: Any model accepting ``ModelInput`` → ``ModelOutput``.
        initial_input: ``ModelInput`` containing the initial sequence
            tensor ``(B, L)`` as ``noisy_token_ids`` (with condition
            prefix filled and maskable positions set to
            ``mask_token_id``) and any per-task metadata (e.g.
            ``drug_indices``, ``dose_values`` for perturbation models).
        mask_token_id: Token id used for masked positions.
        config: Sampler config namespace with fields: ``steps``,
            ``algorithm``, ``t_schedule``, ``t_shift``, ``t_exponent``,
            ``eps``, ``eps_start``, ``enforce_zero_snr``,
            ``unmask_schedule``, ``corrector_cls``,
            ``corrector_strategies``, ``cfg_scale``, ``cfg_step_start``,
            ``cfg_step_end``.
        tokenizer: Tokenizer instance forwarded to the corrector.
        device: Torch device string (e.g. ``"cuda:0"``).
        device_type: Autocast device type (``"cuda"`` / ``"cpu"``).
        dtype: Autocast dtype (e.g. ``torch.bfloat16``).
        cfg_uncond_input_generator: Callable that transforms a conditional
            ``ModelInput`` into an unconditional one for CFG. Required
            when ``cfg_scale > 0``. Must preserve non-condition metadata
            (e.g. ``mask_indices``).
        corrector_sample_gene_names: Gene names for corrector's
            ``restrict_valid_genes`` strategy.
        temperature: Gumbel noise temperature for token sampling.
        print_info: Whether to print sampling diagnostics.

    Returns:
        ``(B, L)`` tensor of sampled token ids.
    """
    # --- Override model's use_lm_head_masked_only if sampler config specifies it ---
    use_lm_head_masked_only = getattr(config, "use_lm_head_masked_only", None)
    if use_lm_head_masked_only is not None and hasattr(model, "config"):
        model.config.use_lm_head_masked_only = bool(use_lm_head_masked_only)

    # --- Read config fields directly ---
    steps = int(config.steps)
    algorithm = str(config.algorithm)
    t_schedule = str(config.t_schedule)
    t_shift = float(config.t_shift)
    t_exponent = float(config.t_exponent)
    eps = float(config.eps)
    eps_start = float(config.eps_start)
    enforce_zero_snr = bool(config.enforce_zero_snr)
    unmask_schedule = str(config.unmask_schedule)
    corrector_cls_name = str(config.corrector_cls)
    corrector_strategies = str(config.corrector_strategies)
    cfg_scale = float(config.cfg_scale)
    cfg_step_start = int(config.cfg_step_start)
    cfg_step_end = int(config.cfg_step_end)

    assert algorithm in ("origin",), f"Unknown algorithm: {algorithm!r}"
    assert unmask_schedule in ("default", "deterministic"), (
        f"Unknown unmask_schedule: {unmask_schedule!r}"
    )

    # --- Unpack ModelInput: x stays as Tensor for high-frequency sampling ops;
    #     base_metadata snapshot holds per-task metadata (drug_indices, etc.)
    #     that is constant across steps. Per-step mask_indices is injected
    #     into step_metadata at the forward boundary only. ---
    x = initial_input.noisy_token_ids.clone()
    base_metadata = dict(initial_input.metadata)
    batch_size, block_size = x.shape

    # --- Timestep schedule ---
    timesteps = get_timesteps(
        t_schedule, steps, eps_start, eps,
        t_shift, t_exponent, enforce_zero_snr,
        device=device,
    )

    # --- Deterministic unmask schedule (optional) ---
    deterministic_unmask_schedule = None
    if unmask_schedule == "deterministic":
        initial_mask_counts = (x == mask_token_id).sum(dim=-1)
        deterministic_unmask_schedule = get_unmask_token_numbers(
            timesteps, initial_mask_counts,
        )

    # --- Corrector ---
    vocab_size = mask_token_id  # mask_token_id == padded_vocab_size == vocab_size
    corrector = build_corrector(
        corrector_cls_name,
        seq_shape=(batch_size, block_size),
        logit_dim=vocab_size,
        tokenizer=tokenizer,
        strategies=corrector_strategies,
        device=device,
        sample_gene_names=corrector_sample_gene_names,
    )

    # --- CFG step range ---
    effective_cfg_end = cfg_step_end if cfg_step_end >= 0 else steps

    if print_info:
        print(
            f"======================= Sampling info =======================\n"
            f"Batch size: {batch_size}\n"
            f"Steps: {steps}\n"
            f"Algorithm: {algorithm}\n"
            f"Temperature: {temperature}\n"
            f"CFG scale: {cfg_scale}\n"
            f"CFG step range: [{cfg_step_start}, {effective_cfg_end})\n"
            f"eps: {eps}, eps_start: {eps_start}\n"
            f"Timestep schedule: {t_schedule}\n"
            f"Timestep shift: {t_shift}\n"
            f"Enforce zero SNR: {enforce_zero_snr}\n"
            f"Block size: {block_size}\n"
            f"Unmask schedule: {unmask_schedule}\n"
            f"Corrector: {corrector_cls_name}, strategies={corrector_strategies}\n"
            f"=============================================================\n"
        )

    # --- Check if model supports masked-only lm_head ---
    use_masked_lm = (
        hasattr(model, "config")
        and getattr(model.config, "use_lm_head_masked_only", False)
    )

    # Pre-allocate lookup table for masked-only logit indexing (reused each step).
    flat_lookup = (
        torch.empty((batch_size, block_size), dtype=torch.long, device=device)
        if use_masked_lm else None
    )

    # --- Main sampling loop ---
    for i in tqdm(range(steps), disable=not print_info):
        mask_index = x == mask_token_id
        num_masked_per_sample = mask_index.sum(dim=-1)

        if num_masked_per_sample.sum() == 0:
            break

        # --- Build per-step ModelInput for forward boundary ---
        # x stays as Tensor for everything else; ModelInput is only constructed
        # here so that forward_model / forward_with_cfg get a unified object
        # carrying both noisy_token_ids and metadata (including mask_indices).
        step_metadata = (
            base_metadata if not use_masked_lm
            else {**base_metadata, "mask_indices": mask_index}
        )
        step_input = ModelInput(noisy_token_ids=x, metadata=step_metadata)

        # --- Model prediction (with optional CFG) ---
        use_cfg = cfg_scale > 0.0 and cfg_step_start <= i < effective_cfg_end

        if use_cfg and cfg_uncond_input_generator is not None:
            logits, is_masked_only = forward_with_cfg(
                model, step_input, cfg_uncond_input_generator, cfg_scale,
                device_type=device_type, dtype=dtype,
            )
        else:
            logits, is_masked_only = forward_model(
                model, step_input, device_type=device_type, dtype=dtype,
            )

        # --- Logit correction (corrector operates on raw Tensor x) ---
        if is_masked_only:
            batch_idx_flat, pos_idx_flat = torch.where(mask_index)
            if corrector is not None:
                logits = corrector.correct_logits(
                    logits, x, masked_positions=pos_idx_flat,
                )
        else:
            if corrector is not None:
                logits = corrector.correct_logits(logits, x)

        # --- Determine number of tokens to unmask ---
        t, s = timesteps[i], timesteps[i + 1]

        if deterministic_unmask_schedule is not None:
            scheduled_k = deterministic_unmask_schedule[i]
            num_to_unmask = torch.minimum(scheduled_k, num_masked_per_sample)
        else:
            p_transfer = 1 - s / t if i < steps - 1 else 1.0
            num_to_unmask = (num_masked_per_sample.float() * p_transfer).ceil().long()

        num_to_unmask = num_to_unmask.clamp(min=0)

        # --- Position selection ---
        if algorithm == "origin":
            selection_scores = torch.rand_like(x, dtype=torch.float, device=device)
        else:
            raise NotImplementedError(f"Algorithm {algorithm!r} is not implemented.")

        selection_scores.masked_fill_(~mask_index, -1.0)

        max_k = num_to_unmask.max().item()
        if max_k == 0:
            continue

        _, top_indices = torch.topk(selection_scores, k=max_k, dim=-1)

        k_mask = (
            torch.arange(max_k, device=device).expand(batch_size, max_k)
            < num_to_unmask.unsqueeze(-1)
        )

        indices_to_unmask = top_indices[k_mask]
        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(-1)
            .expand(batch_size, max_k)[k_mask]
        )

        # --- Token sampling via Gumbel-max ---
        if is_masked_only:
            n_mask = len(batch_idx_flat)
            flat_lookup.fill_(n_mask)
            flat_lookup[batch_idx_flat, pos_idx_flat] = torch.arange(n_mask, device=device)
            logit_sel = flat_lookup[batch_indices, indices_to_unmask]
            sampled_tokens = categorical_sample_by_gumbel_max(
                logits[logit_sel], temperature=temperature,
            )
        else:
            sampled_tokens = categorical_sample_by_gumbel_max(
                logits[batch_indices, indices_to_unmask], temperature=temperature,
            )

        x[batch_indices, indices_to_unmask] = sampled_tokens

    return x
