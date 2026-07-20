"""Reusable workflow helpers shared across train / infer / eval workflows."""

from __future__ import annotations

import glob as glob_module
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from src.modules.common import CheckpointRef
from src.modules.ema.checkpoint_naming import build_ema_glob_pattern
from src.modules.models import SequenceFormatConfig
from src.modules.models.transencoder_dense_patch.config import find_multiple

if TYPE_CHECKING:
    from workflows.train.config_schema import CheckpointRefConfig


def infer_patch_aligned_block_size(
    *,
    seq_length: int,
    condition_prefix_length: int,
    patch_size: int = 1,
    min_tail_pad: int = 1,
    make_even: bool = True,
) -> int:
    """Infer a patch-aligned block_size from sequence layout parameters.

    Port of ``archive/legacy/lit_gpt/utils.py:infer_dense_block_size_from_tokenizer``.
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if min_tail_pad < 0:
        raise ValueError(f"min_tail_pad must be >= 0, got {min_tail_pad}")

    gene_len = seq_length - condition_prefix_length
    if gene_len <= 0:
        raise ValueError(
            f"Invalid gene_len={gene_len} from seq_length={seq_length} "
            f"and condition_prefix_length={condition_prefix_length}"
        )

    pad_for_patch = (-gene_len) % patch_size if patch_size > 1 else 0
    tail_needed = max(min_tail_pad, pad_for_patch)
    block_size = seq_length + tail_needed
    if make_even:
        block_size = find_multiple(block_size, 2)
    return block_size


def resolve_fabric_precision(dtype: str) -> str:
    mapping = {
        "float32": "32-true",
        "float16": "16-mixed",
        "bfloat16": "bf16-mixed",
    }
    return mapping[dtype]


def build_model_sequence_format(tokenizer: object) -> SequenceFormatConfig:
    return SequenceFormatConfig(
        condition_prefix_length=int(tokenizer.condition_prefix_length),
        seq_length=int(tokenizer.seq_length),
        target_gene_high_idx=getattr(tokenizer, "condition_target_gene_high_idx", -1),
        target_gene_low_idx=getattr(tokenizer, "condition_target_gene_low_idx", -1),
        target_gene_token_id_offset=getattr(tokenizer, "target_gene_token_id_offset", None),
        target_gene_2token_width=getattr(tokenizer, "target_gene_2token_width", None),
    )


def instantiate_model(
    model_cfg: DictConfig,
    *,
    sequence_format: SequenceFormatConfig,
    vocab_size: int,
    block_size: int,
) -> object:
    if "config" not in model_cfg:
        raise ValueError("Model config is missing `config`.")
    config_cfg = model_cfg.config
    resolved_model_config = instantiate(
        config_cfg,
        sequence_format=sequence_format,
        vocab_size=vocab_size,
        block_size=block_size,
    )
    model_target = model_cfg.get("_target_")
    if not isinstance(model_target, str) or not model_target.strip():
        raise ValueError("Model config is missing a valid '_target_'.")
    model_cls = get_object(model_target)
    return model_cls(resolved_model_config)


def resolve_mask_token_id(*, model: object) -> int:
    return int(model.config.padded_vocab_size)


def build_checkpoint_ref(cfg: "CheckpointRefConfig") -> CheckpointRef:
    return CheckpointRef(
        path=cfg.path,
        path_type=cfg.path_type,
        resolve_rule=cfg.resolve_rule,
    )


# ------------------------------------------------------------------
# Checkpoint path utilities (single source of truth for naming)
# ------------------------------------------------------------------

CKPT_SUBDIR = "ckpt"
EMA_SUBDIR = "ema"


def build_ckpt_path(ckpt_dir: str | Path, opt_step: int) -> Path:
    """Construct the exact path for a main (non-EMA) checkpoint file.

    Args:
        ckpt_dir: Directory containing main checkpoint files.
        opt_step: Optimizer step number.

    Returns:
        Full path: ``{ckpt_dir}/step-{opt_step:06d}-ckpt.pth``.
    """
    return Path(ckpt_dir) / f"step-{opt_step:06d}-ckpt.pth"


def build_ckpt_glob_pattern(ckpt_dir: str | Path) -> str:
    """Construct a glob pattern matching all main checkpoints in a directory.

    Args:
        ckpt_dir: Directory containing main checkpoint files.

    Returns:
        Glob pattern string: ``{ckpt_dir}/step-*-ckpt.pth``.
    """
    return str(Path(ckpt_dir) / "step-*-ckpt.pth")


def find_ckpt(training_dir: str | Path, iteration: int) -> str | None:
    """Find an existing main checkpoint for the given iteration.

    Args:
        training_dir: Root training output directory (contains ``ckpt/`` subdir).
        iteration: Target optimizer step.

    Returns:
        Resolved path string if found, None otherwise.
    """
    path = build_ckpt_path(Path(training_dir) / CKPT_SUBDIR, iteration)
    return str(path) if path.is_file() else None


def find_ema_ckpt(
    training_dir: str | Path, iteration: int, ema_std: float = 0.050,
) -> str | None:
    """Find an existing EMA checkpoint for the given iteration.

    Args:
        training_dir: Root training output directory (contains ``ema/`` subdir).
        iteration: Target optimizer step.
        ema_std: EMA standard deviation.

    Returns:
        Resolved path string if found, None otherwise. When multiple files
        match (should not happen normally), returns the most recently modified.
    """
    ema_dir = Path(training_dir) / EMA_SUBDIR
    pattern = build_ema_glob_pattern(ema_dir, iteration, ema_std)
    matches = glob_module.glob(pattern)
    if not matches:
        return None
    return max(matches, key=lambda p: Path(p).stat().st_mtime)
