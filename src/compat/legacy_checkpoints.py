"""Legacy checkpoint compatibility layer.

Handles loading checkpoints from the old repository (archived at
``archive/legacy/``) into the refactored model architecture.

Key differences between legacy and new state dicts:

- QKV projection naming: ``attn.attn`` (legacy) → ``attn.qkv`` (new).
- Legacy checkpoints may contain TransformerEngine ``_extra_state`` keys
  (zero-size tensors) that should be preserved when TE is available or
  filtered when it is not.
- Legacy EMA files use ``iter-*`` prefix; new ones use ``step-*``.
- Legacy EMA dict has ``cur_nsamples``/``init_nsamples``; new has
  ``samples_seen``.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class LegacyCheckpointMetadata:
    """Metadata extracted from a legacy checkpoint path/content.

    Attributes:
        source_path: Original file path.
        iteration: Training iteration parsed from filename.
        ema_std: EMA standard deviation (None if not an EMA checkpoint).
        samples_seen: Number of samples seen at checkpoint time.
    """

    source_path: str
    iteration: int
    ema_std: Optional[float]
    samples_seen: int


# Pattern for legacy EMA filenames:
# iter-550000-ema-140800000samples-0.050std.pth
_LEGACY_EMA_PATTERN = re.compile(
    r"iter-(\d+)-ema-(\d+)samples-([\d.]+)std\.pth$"
)


def is_legacy_checkpoint(path: str, raw: Optional[Dict[str, Any]] = None) -> bool:
    """Detect whether a checkpoint file is from the legacy repository.

    Detection heuristics (any one is sufficient):
    1. Filename matches the legacy ``iter-*-ema-*`` pattern.
    2. Loaded dict contains ``cur_nsamples`` key (legacy EMA format).
    3. Loaded dict's ``model`` sub-dict contains ``*.attn.attn.weight`` keys.

    Args:
        path: File path to the checkpoint.
        raw: Optionally pre-loaded checkpoint dict for content inspection.

    Returns:
        True if the checkpoint is identified as legacy format.
    """
    # Heuristic 1: filename pattern.
    if _LEGACY_EMA_PATTERN.search(path):
        return True

    if raw is None:
        return False

    # Heuristic 2: legacy EMA metadata key.
    if isinstance(raw, dict) and "cur_nsamples" in raw:
        return True

    # Heuristic 3: legacy attention key naming.
    state = raw.get("model", raw) if isinstance(raw, dict) else None
    if isinstance(state, dict):
        for key in state:
            if ".attn.attn." in key:
                return True

    return False


def identify_legacy_checkpoint(path: str) -> LegacyCheckpointMetadata:
    """Extract metadata from a legacy checkpoint filename.

    Args:
        path: Path to the legacy checkpoint file.

    Returns:
        Parsed metadata including iteration, EMA std, and samples seen.

    Raises:
        ValueError: If the filename does not match the expected legacy pattern.
    """
    match = _LEGACY_EMA_PATTERN.search(path)
    if not match:
        raise ValueError(
            f"Cannot parse legacy checkpoint metadata from path: {path}"
        )
    return LegacyCheckpointMetadata(
        source_path=path,
        iteration=int(match.group(1)),
        ema_std=float(match.group(3)),
        samples_seen=int(match.group(2)),
    )


def convert_legacy_state_dict(
    state_dict: Dict[str, Any],
    *,
    strict_filter: bool = False,
) -> Dict[str, Any]:
    """Convert a legacy model state dict to the new key naming convention.

    Transformations applied:
    1. Rename ``*.attn.attn.weight`` → ``*.attn.qkv.weight``
       (and similarly for bias if present).
    2. Optionally filter out zero-size TransformerEngine ``_extra_state``
       tensors when ``strict_filter=True``.

    Args:
        state_dict: Legacy model state dict (the ``"model"`` value from
            the checkpoint file).
        strict_filter: If True, remove keys matching
            ``*.te_dot_product_attn._extra_state`` from the output.
            When False (default), these keys are preserved so that models
            with TE enabled can consume them.

    Returns:
        New state dict with remapped keys. The original dict is not modified.
    """
    new_state: Dict[str, Any] = OrderedDict()

    for key, value in state_dict.items():
        new_key = key

        # Remap: transformer.h.X.attn.attn.{weight,bias}
        #      → transformer.h.X.attn.qkv.{weight,bias}
        if ".attn.attn." in key:
            new_key = key.replace(".attn.attn.", ".attn.qkv.")

        # Optionally filter TE extra state.
        if strict_filter and "te_dot_product_attn._extra_state" in new_key:
            continue

        new_state[new_key] = value

    return new_state


def load_legacy_checkpoint(
    path: str,
    *,
    device: Any = "cpu",
) -> Tuple[Dict[str, Any], LegacyCheckpointMetadata]:
    """Load a legacy checkpoint and return the converted state dict + metadata.

    Convenience function combining detection, metadata extraction, loading,
    and state dict conversion.

    Args:
        path: Path to the legacy ``.pth`` checkpoint.
        device: Device to map tensors to during loading.

    Returns:
        Tuple of (converted_state_dict, metadata).

    Raises:
        ValueError: If the file is not a recognized legacy checkpoint.
    """
    import torch

    raw = torch.load(path, map_location=device, weights_only=False)

    if not is_legacy_checkpoint(path, raw):
        raise ValueError(f"File does not appear to be a legacy checkpoint: {path}")

    # Extract metadata from filename (best-effort).
    try:
        metadata = identify_legacy_checkpoint(path)
    except ValueError:
        # Fallback: build metadata from file content.
        metadata = LegacyCheckpointMetadata(
            source_path=path,
            iteration=0,
            ema_std=raw.get("std") if isinstance(raw, dict) else None,
            samples_seen=(
                raw.get("cur_nsamples", 0) if isinstance(raw, dict) else 0
            ),
        )

    # Extract and convert state dict.
    state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
    converted = convert_legacy_state_dict(state_dict)

    return converted, metadata
