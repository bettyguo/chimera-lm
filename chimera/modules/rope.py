"""Rotary positional embedding (Su et al. 2021).

We pre-compute cos/sin tables out to a max position and apply them to Q and K.
For decode at position t, the caller indexes the table at t.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Standard RoPE with a precomputed cos/sin cache.

    Shapes:
      cos, sin:  (max_seq_len, head_dim)
      apply input/output: (..., T, head_dim)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        # Interleave to (max_seq_len, head_dim) by repeat: [f0,f0,f1,f1,...]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE.

        x:         (B, T, H, Dk) or (B, T, Dk) — RoPE rotates the last dim, indexed by T.
        positions: (T,) integer indices into the precomputed tables.
        """
        # register_buffer return type is nn.Module per stubs; values are Tensors.
        cos = self.cos_cached[positions].to(x.dtype)  # type: ignore[index]
        sin = self.sin_cached[positions].to(x.dtype)  # type: ignore[index]
        if x.dim() == 4:
            # cos/sin: (T, Dk) -> (1, T, 1, Dk)
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        else:
            raise ValueError(f"unexpected RoPE input rank {x.dim()}")
        return x * cos + self._rotate_half(x) * sin
