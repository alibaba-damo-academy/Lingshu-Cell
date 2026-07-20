"""Model interfaces and direct model/config exports."""

from __future__ import annotations

from typing import Type
from .base import Model
from .transencoder_dense_patch.config import SequenceFormatConfig, TransEncoderConfig
from .transencoder_dense_patch.model import TransEncoder


MODEL_REGISTRY: dict[str, Type[Model]] = {
    "patch_dense": TransEncoder,
}

__all__ = [
    "Model",
    "MODEL_REGISTRY",
    "SequenceFormatConfig",
    "TransEncoder",
    "TransEncoderConfig",
]
