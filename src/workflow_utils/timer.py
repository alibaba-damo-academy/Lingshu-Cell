"""Reusable timer for profiling named code sections."""

from __future__ import annotations

import time
from typing import Optional

import torch


class Timer:
    """Accumulating timer for named code sections.

    Supports optional CUDA synchronization for accurate GPU timing.

    Usage::

        timer = Timer(enabled=True, cuda_sync=True)

        timer.start("forward_backward")
        # ... model forward + backward ...
        timer.stop()

        timer.start("optimizer")
        # ... optimizer step ...
        timer.stop()

        metrics = timer.get_and_reset()
        # {"timing/forward_backward_ms": 28.5, "timing/optimizer_ms": 3.2}

    Args:
        enabled: When False, all operations are no-ops (zero overhead).
        cuda_sync: When True, calls ``torch.cuda.synchronize()`` at
            start/stop boundaries for accurate GPU timing.
    """

    def __init__(self, enabled: bool = True, cuda_sync: bool = False) -> None:
        self._enabled = enabled
        self._cuda_sync = cuda_sync
        self._accum: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._current_name: Optional[str] = None
        self._current_start: float = 0.0

    def start(self, name: str) -> None:
        """Begin timing a named section."""
        if not self._enabled:
            return
        if self._cuda_sync:
            torch.cuda.synchronize()
        self._current_name = name
        self._current_start = time.perf_counter()

    def stop(self) -> None:
        """End timing the current section."""
        if not self._enabled:
            return
        if self._cuda_sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._current_start
        name = self._current_name
        self._accum[name] = self._accum.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def get_and_reset(self) -> dict[str, float]:
        """Return average milliseconds per section and reset accumulators.

        Returns:
            Dict with keys ``"timing/{name}_ms"`` → average ms.
        """
        if not self._enabled:
            return {}
        results = {}
        for name in self._accum:
            count = self._counts[name]
            if count > 0:
                results[f"timing/{name}_ms"] = self._accum[name] / count * 1000
        self._accum.clear()
        self._counts.clear()
        return results
