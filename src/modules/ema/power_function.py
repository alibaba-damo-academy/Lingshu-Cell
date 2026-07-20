from __future__ import annotations

import glob as glob_module
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .checkpoint_naming import build_ema_ckpt_path, build_ema_glob_pattern
from ..models.base import Model
from .base import EMA


class PowerFunctionEMA(EMA):
    """
    Power-function EMA for the VCC train path.

    Legacy counterpart: `archive/legacy/lit_gpt/phema.py:PowerFunctionEMA`
    """

    def __init__(self, *, model: Model, enabled: bool, stds: Sequence[float]) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.stds = tuple(float(std) for std in stds)
        self.shadow_params: list[dict[str, torch.Tensor]] = [dict() for _ in self.stds]
        self.model_params: dict[str, torch.nn.Parameter] = {}

    def register(self, model: Model) -> None:
        if not self.enabled:
            return
        self.model = model
        self.model_params = {}
        self.shadow_params = [dict() for _ in self.stds]
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.model_params[name] = parameter
                for shadow in self.shadow_params:
                    shadow[name] = parameter.detach().clone()

    def update(self, *, current_num_samples: int, batch_size: int) -> None:
        if not self.enabled:
            return
        if current_num_samples <= 0 or batch_size <= 0:
            return
        with torch.no_grad():
            for std, shadow in zip(self.stds, self.shadow_params):
                beta = float(power_function_beta(std=std, t_next=current_num_samples, t_delta=batch_size))
                for name, shadow_parameter in shadow.items():
                    shadow_parameter.lerp_(self.model_params[name].detach(), 1.0 - beta)

    # ------------------------------------------------------------------
    # Checkpoint save / load — FSDP-compatible via backup_and_restore
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_ckpt_path(
        ema_dir: Path, opt_step: int, samples_seen: int, std: float,
    ) -> Path:
        return build_ema_ckpt_path(ema_dir, opt_step, samples_seen, std)

    @staticmethod
    def _find_ema_ckpt(ema_dir: Path, opt_step: int, std: float) -> Path | None:
        pattern = build_ema_glob_pattern(ema_dir, opt_step, std)
        matches = glob_module.glob(pattern)
        if not matches:
            return None
        return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))

    @contextmanager
    def backup_and_restore(self):
        """Back up live model params (to CPU for CUDA tensors), yield, then restore."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in self.model_params.items():
            p_data = param.data
            if p_data.device.type == "cuda":
                backup[name] = p_data.detach().to(device="cpu", copy=True)
            else:
                backup[name] = p_data.clone()
        try:
            yield
        finally:
            for name, param in self.model_params.items():
                b = backup[name]
                if b.device == param.data.device:
                    param.data.copy_(b)
                else:
                    param.data.copy_(b.to(device=param.data.device, non_blocking=False))

    def _apply_ema(self, std: float) -> None:
        """Copy shadow weights for *std* into the live model parameters."""
        idx = self.stds.index(std)
        for name, param in self.model_params.items():
            param.data.copy_(self.shadow_params[idx][name])

    def _load_ema_from_model(self, model: Any, std: float) -> None:
        """Copy current model parameters into the shadow slot for *std*."""
        idx = self.stds.index(std)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[idx][name] = param.data.clone()

    def save_checkpoint(
        self,
        *,
        fabric: Any,
        model: Any,
        ema_dir: str | Path,
        opt_step: int,
        samples_seen: int,
    ) -> None:
        if not self.enabled:
            return
        ema_dir = Path(ema_dir)
        ema_dir.mkdir(parents=True, exist_ok=True)
        with self.backup_and_restore():
            for std in self.stds:
                fabric.barrier()
                self._apply_ema(std)
                ema_path = self._ema_ckpt_path(ema_dir, opt_step, samples_seen, std)
                ema_state = {"model": model, "std": std, "samples_seen": samples_seen}
                fabric.save(str(ema_path), ema_state)
                del ema_state

    def load_checkpoint(
        self,
        *,
        fabric: Any,
        model: Any,
        ema_dir: str | Path,
        opt_step: int,
    ) -> bool:
        if not self.enabled:
            return False
        ema_dir = Path(ema_dir)
        if not ema_dir.is_dir():
            return False
        loaded_any = False
        with self.backup_and_restore():
            for std in self.stds:
                ema_path = self._find_ema_ckpt(ema_dir, opt_step, std)
                if ema_path is None:
                    continue
                ema_state = {"model": model, "std": std, "samples_seen": 0}
                fabric.load(str(ema_path), ema_state)
                self._load_ema_from_model(model, std)
                loaded_any = True
                del ema_state
        return loaded_any

    # ------------------------------------------------------------------
    # state_dict / load_state_dict — kept for non-FSDP or diagnostic use
    # ------------------------------------------------------------------

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "stds": list(self.stds),
            "shadow_params": [
                {name: parameter.clone() for name, parameter in shadow.items()}
                for shadow in self.shadow_params
            ],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        loaded_stds = state_dict.get("stds")
        if loaded_stds is not None:
            self.stds = tuple(float(std) for std in loaded_stds)
        self.shadow_params = [dict() for _ in self.stds]
        for shadow, loaded_shadow in zip(self.shadow_params, state_dict.get("shadow_params", [])):
            for name, tensor in loaded_shadow.items():
                shadow[name] = torch.as_tensor(tensor).clone()


def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp


def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta
