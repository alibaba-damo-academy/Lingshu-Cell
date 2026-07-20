"""EMA checkpoint filename conventions (single source of truth).

Pure functions with no heavy dependencies — safe to import from anywhere.
"""

from __future__ import annotations

from pathlib import Path


def build_ema_ckpt_path(
    ema_dir: str | Path, opt_step: int, samples_seen: int, std: float,
) -> Path:
    """Construct the exact path for an EMA checkpoint file.

    Args:
        ema_dir: Directory containing EMA checkpoint files.
        opt_step: Optimizer step number.
        samples_seen: Total samples processed at this step.
        std: EMA standard deviation.

    Returns:
        Full path: ``{ema_dir}/step-{opt_step:06d}-ema-{samples_seen:010d}samples-{std:.3f}std.pth``.
    """
    return Path(ema_dir) / f"step-{opt_step:06d}-ema-{samples_seen:010d}samples-{std:.3f}std.pth"


def build_ema_glob_pattern(
    ema_dir: str | Path, opt_step: int, std: float,
) -> str:
    """Construct a glob pattern to find an EMA checkpoint for a given step.

    Args:
        ema_dir: Directory containing EMA checkpoint files.
        opt_step: Optimizer step number.
        std: EMA standard deviation.

    Returns:
        Glob pattern string (samples_seen part uses wildcard).
    """
    return str(Path(ema_dir) / f"step-{opt_step:06d}-ema-*samples-{std:.3f}std.pth")
