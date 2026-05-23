"""Tests for the PureTransformerLM and PureSSMLM baselines."""

import torch

from chimera.baselines import BaselineConfig, PureSSMLM, PureTransformerLM


def test_transformer_baseline_forward_shape():
    cfg = BaselineConfig(vocab_size=64, num_layers=2, dim=32, num_heads=4, max_seq_len=64)
    model = PureTransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model.forward_prefill(ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.router_outputs == []  # baselines have no router


def test_ssm_baseline_forward_shape():
    cfg = BaselineConfig(vocab_size=64, num_layers=2, dim=32, num_heads=4, max_seq_len=64)
    model = PureSSMLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model.forward_prefill(ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)


def test_transformer_baseline_causal():
    """Perturbing future input must not change earlier output logits."""
    cfg = BaselineConfig(vocab_size=64, num_layers=2, dim=32, num_heads=4, max_seq_len=32)
    model = PureTransformerLM(cfg).double()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    out_a = model.forward_prefill(ids).logits

    ids_b = ids.clone()
    ids_b[:, 6:] = (ids_b[:, 6:] + 5) % cfg.vocab_size
    out_b = model.forward_prefill(ids_b).logits
    # First 6 positions must be identical.
    assert torch.allclose(out_a[:, :6], out_b[:, :6], atol=1e-12)
    # And the perturbed range should differ.
    assert not torch.allclose(out_a[:, 6:], out_b[:, 6:], atol=1e-6)


def test_baselines_have_same_param_count_at_same_dim():
    """Same (L, D, H) => Transformer and SSM should have comparable param counts.
    They differ only by the mixer subblock; everything else is shared."""
    cfg = BaselineConfig(vocab_size=64, num_layers=4, dim=64, num_heads=4, max_seq_len=64)
    t = PureTransformerLM(cfg)
    s = PureSSMLM(cfg)
    t_params = sum(p.numel() for p in t.parameters())
    s_params = sum(p.numel() for p in s.parameters())
    # FFN dominates; transformer has Q/K/V/O which is 4 d^2 = 16K, SSM has
    # ssm_in + ssm_out + decay which is ~2*d*S + S = ~4K with S=32. Transformer
    # will be heavier by ~per-layer 12K * num_layers.
    # Just sanity-check both are in the same ballpark (<3x ratio).
    ratio = max(t_params, s_params) / min(t_params, s_params)
    assert ratio < 3, f"baselines differ too much: {t_params} vs {s_params}"
