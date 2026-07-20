"""Timestep schedule and unmask planning utilities for reverse masked diffusion.

Ported from the authoritative batch sampler at
``TransMDM_VCC/eval/batch_default_sampler.py`` (L26-163). Semantics are
preserved exactly; only the module boundary changed.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise for categorical sampling via the Gumbel-max trick.

    Uses float64 as recommended by https://arxiv.org/pdf/2409.02908.

    Args:
        logits: Unnormalized log-probabilities of arbitrary shape.
        temperature: Noise temperature controlling sampling sharpness.

    Returns:
        Noised logits in the same shape, computed as ``exp(logits) / (-log(u))^temperature``.
    """
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def categorical_sample_by_gumbel_max(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample from categorical distribution using the Gumbel-max trick.

    Args:
        logits: Unnormalized log-probabilities, shape ``(..., V)``.
        temperature: Gumbel noise temperature.

    Returns:
        Sampled token indices, shape ``(...)``.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    return torch.argmax(logits_with_noise, dim=-1)


def logsnr_shift(timesteps: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Apply log-SNR shift to timesteps.

    Transforms ``t`` via ``shift * t / ((shift - 1) * t + 1)``, which
    uniformly distributes steps in log-SNR space when ``shift > 1``.

    Args:
        timesteps: Timestep values in ``[0, 1]``.
        shift: Shift factor (``snr = (1 - t) / t``).

    Returns:
        Shifted timesteps in the same shape.
    """
    return shift * timesteps / ((shift - 1) * timesteps + 1)


def get_timesteps(
    t_schedule: str,
    steps: int,
    eps_start: float,
    eps_end: float,
    t_shift: float = 1.0,
    t_exponent: float = 1.0,
    enforce_zero_snr: bool = False,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a monotonically decreasing timestep schedule.

    Args:
        t_schedule: Schedule type — ``"linear-t"``, ``"linear-log-snr"``,
            or ``"cosine-t"``.
        steps: Number of sampling steps (``S``).
        eps_start: Small value near ``t=1`` end (used by ``linear-log-snr``).
        eps_end: Small value near ``t=0`` end.
        t_shift: Log-SNR shift factor applied after schedule generation.
        t_exponent: Power applied to timesteps before the shift.
        enforce_zero_snr: If ``True``, forces the first timestep to
            exactly ``1.0`` (zero SNR) for ``linear-log-snr``.
        device: Torch device for the output tensor.

    Returns:
        1-D tensor of shape ``(S + 1,)`` with timesteps decreasing from
        near ``1.0`` to near ``0.0``.
    """
    if t_schedule == "linear-t":
        timesteps = torch.linspace(1, eps_end, steps + 1, device=device)
    elif t_schedule == "linear-log-snr":
        logsnr_T = math.log(eps_start / (1 - eps_start))
        logsnr_eps = math.log((1 - eps_end) / eps_end)
        if enforce_zero_snr:
            logsnrs = torch.linspace(logsnr_T, logsnr_eps, steps, device=device)
            timesteps = 1 / (logsnrs.exp() + 1)
            timesteps = torch.cat([timesteps.new_ones([1]), timesteps], dim=0)
        else:
            logsnrs = torch.linspace(logsnr_T, logsnr_eps, steps + 1, device=device)
            timesteps = 1 / (logsnrs.exp() + 1)
    elif t_schedule == "cosine-t":
        timesteps = torch.cos(
            torch.linspace(0, math.pi / 2, steps + 1, device=device)
        ).clamp(min=0, max=1)
    else:
        raise NotImplementedError(f"t_schedule {t_schedule!r} is not implemented.")

    if t_exponent != 1:
        timesteps = (timesteps ** t_exponent).clamp(min=0, max=1)
    if t_shift != 1:
        timesteps = logsnr_shift(timesteps, shift=t_shift).clamp(min=0, max=1)

    return timesteps


def _generate_single_schedule(
    total_tokens: int,
    unmask_proportions: torch.Tensor,
) -> torch.Tensor:
    """Generate a deterministic unmasking schedule for one sequence.

    Uses the largest-remainder method to distribute integer token counts
    across steps while preserving the overall curve shape.

    Args:
        total_tokens: Total number of tokens to unmask for this sequence.
        unmask_proportions: Ideal proportion of tokens to unmask at each
            step, shape ``(S,)``. Must sum to ``1.0``.

    Returns:
        1-D integer tensor of shape ``(S,)`` with per-step unmask counts
        that sum to ``total_tokens``.
    """
    steps = unmask_proportions.shape[0]

    if steps >= total_tokens:
        unmask_schedule_int = torch.zeros(steps, dtype=torch.long, device=unmask_proportions.device)
        unmask_schedule_int[:total_tokens] = 1
        return unmask_schedule_int

    # Proportional allocation with largest-remainder rounding.
    ideal_schedule_float = unmask_proportions * total_tokens
    unmask_schedule_int = ideal_schedule_float.floor().long()

    remainder = total_tokens - unmask_schedule_int.sum()
    if remainder > 0:
        residuals = ideal_schedule_float - unmask_schedule_int.float()
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        unmask_schedule_int[top_indices] += 1

    assert unmask_schedule_int.sum() == total_tokens, "Internal sum check failed after rounding."

    # Borrow-and-lend adjustment to enforce at-least-one-per-step.
    zero_mask = unmask_schedule_int == 0
    num_zeros = zero_mask.sum().item()

    if num_zeros > 0:
        unmask_schedule_int[zero_mask] = 1
        num_to_borrow = num_zeros
        while num_to_borrow > 0:
            eligible_donors_mask = unmask_schedule_int > 1
            if not eligible_donors_mask.any():
                raise RuntimeError("Cannot enforce min=1 constraint; not enough tokens to distribute.")
            max_val = unmask_schedule_int[eligible_donors_mask].max()
            potential_indices = torch.where((unmask_schedule_int == max_val) & eligible_donors_mask)[0]
            idx_to_borrow_from = potential_indices[0]
            unmask_schedule_int[idx_to_borrow_from] -= 1
            num_to_borrow -= 1

    return unmask_schedule_int


def get_unmask_token_numbers(
    timesteps: torch.Tensor,
    num_tokens_per_sample: Union[torch.LongTensor, torch.Tensor],
) -> torch.Tensor:
    """Generate a deterministic per-step unmask schedule for a batch.

    Args:
        timesteps: 1-D tensor of shape ``(S + 1,)`` defining the decay
            curve (from near ``1.0`` to near ``0.0``).
        num_tokens_per_sample: 1-D tensor of shape ``(B,)`` where each
            element is the total number of masked tokens for that sample.

    Returns:
        2-D integer tensor of shape ``(S, B)`` where each column is the
        per-step unmask schedule for one sample.
    """
    steps = timesteps.shape[0] - 1

    ideal_timesteps, _ = torch.sort(timesteps, descending=True)
    unmask_proportions = ideal_timesteps[:-1] - ideal_timesteps[1:]
    unmask_proportions = unmask_proportions / unmask_proportions.sum()

    schedules = [
        _generate_single_schedule(n.item(), unmask_proportions)
        for n in num_tokens_per_sample
    ]

    final_schedule = torch.stack(schedules, dim=1)

    assert (final_schedule.sum(dim=0) == num_tokens_per_sample).all(), "Final sum check failed for batch."
    enough_tokens_mask = num_tokens_per_sample >= steps
    if enough_tokens_mask.any():
        assert (final_schedule[:, enough_tokens_mask] >= 1).all(), "Final min=1 check failed."

    return final_schedule
