from __future__ import annotations

from typing import Any, Iterable, Protocol

from ..common.interfaces import ModelInput, ModelOutput


class Model(Protocol):
    """
    Minimal train-side model boundary for the Stage 3 VCC mainline.

    The train workflow assembles the model between forward diffusion and loss:
    `ModelInput.noisy_token_ids -> ModelOutput.logits`. Concrete implementations
    are expected to remain compatible with `torch.nn.Module` semantics so the
    workflow can pass them through Fabric setup, optimizer construction, and
    checkpoint save/load boundaries.

    Current stable workflow-facing attributes / methods:
    - `config`
      - workflow reads `model.config.padded_vocab_size` when deriving the mask token id
    - `parameters()`
      - workflow collects trainable parameters for optimizer construction
    - `named_parameters()`
      - EMA registration iterates named trainable parameters
    """

    config: Any

    def parameters(self) -> Iterable[Any]:
        """Yield model parameters for optimizer construction."""

    def named_parameters(self) -> Iterable[tuple[str, Any]]:
        """Yield named model parameters for EMA registration and related runtime bookkeeping."""

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """
        Run one forward pass from normalized train-side model inputs.

        Args:
            inputs: Normalized model input bundle. The current train mainline
                only requires `inputs.noisy_token_ids`, typically a batched
                tensor aligned like `B x L` after forward diffusion/noising.

        Returns:
            `ModelOutput` whose `logits` field is expected to align with the
            original token positions and vocabulary axis, typically `B x L x V`,
            so the downstream loss can compare it against the target
            `LossInputContext.token_ids`.
        """
