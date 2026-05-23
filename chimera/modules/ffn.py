"""SwiGLU feed-forward network.

y = W_o (SiLU(W_gate x) * W_up x)

The hidden dim is rounded up to a multiple of `multiple_of` to keep matmul shapes
hardware-friendly. Default expansion (`mult * d` with mult=8/3) matches Llama.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU MLP.

    Shapes:
      x:    (..., D)
      out:  (..., D)
    """

    def __init__(self, dim: int, hidden_mult: float = 8 / 3, multiple_of: int = 64) -> None:
        super().__init__()
        raw_hidden = int(dim * hidden_mult)
        hidden = ((raw_hidden + multiple_of - 1) // multiple_of) * multiple_of
        self.dim = dim
        self.hidden = hidden
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
