"""SSM mode for CHIMERA — currently a toy first-order recurrence.

Production CHIMERA uses Mamba-2 SSD from the `mamba-ssm` package. That package
requires CUDA + a working Triton install and won't build on Windows / CPU. We
therefore expose a `ToySSM` here with the *same interface* the production wrapper
will satisfy, so swapping in Mamba-2 is a Phase-1.5 wiring change:

    forward_prefill(x: (B,T,D)) -> (y: (B,T,D), final_state: (B,S))
    step(x_t: (B,D), prior_state: (B,S)) -> (y_t: (B,D), new_state: (B,S))

Toy recurrence:
    new_state = sigmoid(decay) * prior_state + W_in @ x_t
    y_t       = W_out @ new_state

The output uses the *post-update* state, which is what mode-1 read should
return at decode (per spec A.2: "mode 1: o_t = SSM_read(s_{t-1}, x_t); will
also produce new state s_t").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToySSM(nn.Module):
    """First-order linear recurrence.

    Not the production SSM — Mamba-2 is selective (data-dependent decay and
    gating). This stand-in matches the *interface* needed for cache parity
    testing and architectural composition.

    Shapes:
      x:           (B, T, D)
      state:       (B, S)
      y:           (B, T, D)
    """

    def __init__(self, dim: int, state_size: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        # decay initialized so sigmoid(.) ≈ 0.9 — long memory by default
        self.decay = nn.Parameter(torch.full((state_size,), 2.197))  # sigmoid(2.197)≈0.9
        self.in_proj = nn.Linear(dim, state_size, bias=False)
        self.out_proj = nn.Linear(state_size, dim, bias=False)

    def empty_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.state_size, device=device, dtype=dtype)

    def step(
        self, x_t: torch.Tensor, prior_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode step.

        x_t:         (B, D)
        prior_state: (B, S)
        returns:     ((B, D) output using post-update state, (B, S) new_state)
        """
        decay = torch.sigmoid(self.decay)  # (S,)
        new_state = decay.unsqueeze(0) * prior_state + self.in_proj(x_t)
        y_t = self.out_proj(new_state)
        return y_t, new_state

    def forward_prefill(
        self, x: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole-sequence forward (used during prefill / training).

        x: (B, T, D); initial_state: (B, S) or None.
        Returns (y: (B, T, D), final_state: (B, S)).

        Implementation: sequential loop. Production (Mamba-2 SSD) uses a parallel
        chunked scan; the *outputs* are identical to the loop because the
        recurrence is associative.
        """
        B, T, _ = x.shape
        if initial_state is None:
            state = self.empty_state(B, device=x.device, dtype=x.dtype)
        else:
            state = initial_state
        outs = []
        for t in range(T):
            y_t, state = self.step(x[:, t], state)
            outs.append(y_t)
        return torch.stack(outs, dim=1), state
