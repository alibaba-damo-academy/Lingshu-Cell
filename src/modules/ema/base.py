from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from ..models.base import Model


class EMA(Protocol):
    """
    Minimal EMA boundary for the train workflow.

    Stable runtime contract:
    - `register(...)` is called after Fabric setup and any checkpoint restore.
    - `update(...)` is the only workflow-driven runtime update entrypoint.
    - `save_checkpoint(...)` / `load_checkpoint(...)` handle per-std EMA
      persistence through Fabric, independent of the main checkpoint.
    - `state_dict/load_state_dict` remain available but are not used for
      FSDP-compatible persistence (use save/load_checkpoint instead).
    """

    def register(self, model: Model) -> None:
        """Bind or re-bind EMA state to the runtime model object."""

    def update(self, *, current_num_samples: int, batch_size: int) -> None:
        """Update EMA state using sample-count progress from the train loop."""

    def save_checkpoint(
        self,
        *,
        fabric: Any,
        model: Any,
        ema_dir: str | Path,
        opt_step: int,
        samples_seen: int,
    ) -> None:
        """Save EMA shadow weights as per-std checkpoint files via Fabric."""

    def load_checkpoint(
        self,
        *,
        fabric: Any,
        model: Any,
        ema_dir: str | Path,
        opt_step: int,
    ) -> bool:
        """Load EMA shadow weights from per-std checkpoint files. Returns True if any std was loaded."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return EMA state."""

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore EMA state."""
