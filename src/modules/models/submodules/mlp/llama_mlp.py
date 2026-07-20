from __future__ import annotations

import torch

try:  # pragma: no cover - optional dependency
    from xformers.ops import SwiGLU
except Exception:  # pragma: no cover - optional dependency
    SwiGLU = None


class GptNeoxMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.proj = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.nn.functional.gelu(self.fc(x)))


class LLaMAMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias
        if SwiGLU is not None:
            self.swiglu = SwiGLU(hidden_size, intermediate_size, bias=bias, _pack_weights=False)
        else:
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "swiglu"):
            return self.swiglu(x)
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
