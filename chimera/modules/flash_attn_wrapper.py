"""flash-attn 2.6+ wrapper — production attention backend on CUDA.

Import-gated like `mamba2_wrapper`. Interface mirrors
`chimera/modules/attention.py`: causal-full and sliding-window variants.

The flash-attn 2.6 API exposes `flash_attn_func` (causal) and `flash_attn_varlen_func`.
For sliding-window, flash-attn 2.6 supports a `window_size=(left, right)` arg
which is exactly what we need (left=W-1, right=0 for causal SWA).

This wrapper is **untested on this dev box** (no CUDA). Verify with
`tests/test_block_shape.py` and `tests/test_causal_consistency.py` after the
GPU swap (those tests are backend-agnostic).
"""

from __future__ import annotations

from typing import Any

import torch


def _ensure_flash_attn_available() -> Any:
    try:
        import flash_attn  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "flash-attn is not installed. Install with `pip install -e \".[gpu]\"` "
            "on a CUDA-capable Linux box. Requires CUDA + nvcc + compatible "
            "flash-attn (>=2.6)."
        ) from e
    return flash_attn


def flash_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Drop-in replacement for `chimera.modules.attention.causal_attention`.

    Shapes:
      q, k, v:  (B, T, H, Dk) — same as the reference impl.
      out:      (B, T, H, Dk)

    flash-attn expects (B, T, H, Dk) directly (no transpose). The function
    handles head-major internally.
    """
    fa = _ensure_flash_attn_available()
    from flash_attn import flash_attn_func  # type: ignore[import-not-found]

    # flash_attn_func signature: q, k, v of shape (B, T, H, Dk) — match ours.
    return flash_attn_func(q, k, v, causal=True)


def flash_sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    """Drop-in replacement for `sliding_window_attention`.

    flash-attn 2.6's `window_size=(left, right)` exposes exact sliding-window
    causal attention. For mode 2 we want left = window-1 (lookback), right=0.
    """
    fa = _ensure_flash_attn_available()
    from flash_attn import flash_attn_func  # type: ignore[import-not-found]

    return flash_attn_func(q, k, v, causal=True, window_size=(window - 1, 0))


def flash_step_attention(
    q_t: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Single-query attention; replacement for `step_attention`.

    flash-attn does support variable-length kernels (`flash_attn_varlen_func`)
    but for a single query the launch overhead dominates. At decode time on
    GPU, prefer cuDNN's scaled-dot-product or batch step-decodes through
    `flash_attn_with_kvcache`. We sketch the simpler path here.

    Shapes:
      q_t:   (B, H, Dk)
      k, v:  (B, Twin, H, Dk)
      out:   (B, H, Dk)
    """
    # Treat q_t as a length-1 sequence and dispatch to flash_attn_func.
    fa = _ensure_flash_attn_available()
    from flash_attn import flash_attn_func  # type: ignore[import-not-found]

    q = q_t.unsqueeze(1)  # (B, 1, H, Dk)
    out = flash_attn_func(q, k, v, causal=True)  # (B, 1, H, Dk)
    return out.squeeze(1)
