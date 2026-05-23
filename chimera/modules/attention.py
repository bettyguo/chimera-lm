"""Causal and sliding-window attention — pure PyTorch.

Production CHIMERA uses flash-attn 2.6's `block_mask` variant for both modes 2
(sliding-window) and 3 (full causal). Flash-attn is a CUDA-only dep, so on
CPU/Windows we substitute a vanilla scaled-dot-product implementation with
explicit masks. The *interfaces* match what the flash-attn wrapper will need:

    causal_attention(q, k, v)               — full causal
    sliding_window_attention(q, k, v, W)    — band-limited causal

Single-step decode versions take (q_t, k_window, v_window) — the caller assembles
the window from the cache (see `chimera.cache.ChimeraCacheLayer.view_with_current`).

Shape convention: multi-head with (B, T, H, Dk). Q/K/V are produced by the
ChimeraBlock projections; this module is purely the attention kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float | None = None
) -> torch.Tensor:
    """Standard causal self-attention.

    Shapes:
      q, k, v:  (B, T, H, Dk)
      out:      (B, T, H, Dk)
    """
    B, T, H, Dk = q.shape
    s = scale if scale is not None else 1.0 / math.sqrt(Dk)
    # (B, H, T, Dk)
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * s  # (B, H, T, T)
    # Causal mask: position i can see j iff j <= i.
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, NEG_INF)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, vh)  # (B, H, T, Dk)
    return out.transpose(1, 2).contiguous()


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention with a per-token lookback of `window` tokens (inclusive).

    Position i attends to j ∈ [max(0, i - window + 1), i].

    Shapes:
      q, k, v:  (B, T, H, Dk)
      out:      (B, T, H, Dk)
    """
    B, T, H, Dk = q.shape
    s = scale if scale is not None else 1.0 / math.sqrt(Dk)
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * s  # (B, H, T, T)
    # Band mask: j <= i AND i - j < window.
    i = torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
    j = torch.arange(T, device=q.device).unsqueeze(0)  # (1, T)
    mask = (j <= i) & ((i - j) < window)
    scores = scores.masked_fill(~mask, NEG_INF)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, vh)
    return out.transpose(1, 2).contiguous()


def step_attention(q_t: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
    """Single-query attention over a key/value window.

    Used at decode for both mode-2 and mode-3 reads. The caller has already
    assembled the appropriate window (sliding ring or persistent cache) and
    *included* the current token's (k_t, v_t).

    Shapes:
      q_t:  (B, H, Dk)
      k, v: (B, Twin, H, Dk)
      out:  (B, H, Dk)
    """
    B, H, Dk = q_t.shape
    s = scale if scale is not None else 1.0 / math.sqrt(Dk)
    # (B, H, 1, Dk) @ (B, H, Dk, Twin) -> (B, H, 1, Twin)
    kh = k.permute(0, 2, 3, 1)  # (B, H, Dk, Twin)
    qh = q_t.unsqueeze(2)        # (B, H, 1, Dk)
    scores = torch.matmul(qh, kh) * s
    weights = F.softmax(scores, dim=-1)  # (B, H, 1, Twin)
    vh = v.permute(0, 2, 1, 3)   # (B, H, Twin, Dk)
    out = torch.matmul(weights, vh).squeeze(2)  # (B, H, Dk)
    return out
