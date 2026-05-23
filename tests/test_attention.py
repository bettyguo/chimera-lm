"""Attention kernel tests — causality, masking, sliding-window correctness."""

import torch

from chimera.modules.attention import (
    causal_attention,
    sliding_window_attention,
    step_attention,
)


def test_causal_attention_doesnt_leak_future():
    """Mutating x[i+1:] must not change output at position i."""
    torch.manual_seed(0)
    B, T, H, Dk = 1, 8, 2, 4
    q = torch.randn(B, T, H, Dk, dtype=torch.float64)
    k = torch.randn(B, T, H, Dk, dtype=torch.float64)
    v = torch.randn(B, T, H, Dk, dtype=torch.float64)
    out1 = causal_attention(q, k, v)
    # Perturb everything from position 4 onward; positions 0..3 must not change.
    k2 = k.clone()
    v2 = v.clone()
    k2[:, 4:] += 1.0
    v2[:, 4:] += 1.0
    out2 = causal_attention(q, k2, v2)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-12)
    # And position 4 *does* change.
    assert not torch.allclose(out1[:, 4], out2[:, 4], atol=1e-6)


def test_sliding_window_locality():
    """For window=W, perturbing position j > i should not change position i.
    Perturbing position j with i - j >= W should not change position i."""
    torch.manual_seed(0)
    B, T, H, Dk = 1, 12, 2, 4
    W = 4
    q = torch.randn(B, T, H, Dk, dtype=torch.float64)
    k = torch.randn(B, T, H, Dk, dtype=torch.float64)
    v = torch.randn(B, T, H, Dk, dtype=torch.float64)
    out1 = sliding_window_attention(q, k, v, W)
    # Perturb position 0 only.
    k2 = k.clone()
    v2 = v.clone()
    k2[:, 0] += 1.0
    v2[:, 0] += 1.0
    out2 = sliding_window_attention(q, k2, v2, W)
    # Position W=4 and beyond shouldn't see position 0 (since i - j = i >= W=4).
    assert torch.allclose(out1[:, W:], out2[:, W:], atol=1e-12)
    # Positions 0..W-1 do see position 0.
    assert not torch.allclose(out1[:, :W], out2[:, :W], atol=1e-6)


def test_step_attention_matches_causal_at_last_position():
    """For a non-cached query at the final position, step-attention should
    match causal-attention's last output."""
    torch.manual_seed(0)
    B, T, H, Dk = 2, 6, 2, 4
    q = torch.randn(B, T, H, Dk, dtype=torch.float64)
    k = torch.randn(B, T, H, Dk, dtype=torch.float64)
    v = torch.randn(B, T, H, Dk, dtype=torch.float64)
    full = causal_attention(q, k, v)
    step = step_attention(q[:, -1], k, v)  # passes full K/V window
    assert torch.allclose(full[:, -1], step, atol=1e-12)
