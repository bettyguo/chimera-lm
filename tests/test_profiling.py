"""Sanity tests for the FLOP/memory accounting formulas in spec A.4."""

from chimera.model import ChimeraConfig
from chimera.utils.profiling import (
    block_flops_per_token_decode,
    block_kv_memory_bytes,
    dense_transformer_flops_per_token,
    dense_transformer_kv_memory_bytes,
)


def test_headline_decode_memory_chimera_smaller_than_dense():
    """Spec A.4 headline: at T=32K with target dist (0.1, 0.6, 0.2, 0.1),
    CHIMERA per-layer KV memory should be ~10x smaller than dense."""
    cfg = ChimeraConfig(
        vocab_size=32_000,
        num_layers=24,
        dim=2048,
        num_heads=16,
        window=512,
        ssm_state=128,
        max_seq_len=32_768,
    )
    T = 32_768
    fractions = (0.10, 0.60, 0.20, 0.10)
    chimera = block_kv_memory_bytes(cfg, fractions, T)
    dense = dense_transformer_kv_memory_bytes(cfg, T)
    # Spec says ~10x; accept anything > 5x as a sanity check.
    ratio = dense / chimera["total"]
    assert ratio > 5, f"expected >5x reduction, got {ratio:.2f}x"
    # Headline number in the spec is ~10x; allow generous tolerance.
    assert 7 < ratio < 15


def test_decode_flops_decrease_with_lower_mode3_fraction():
    cfg = ChimeraConfig(num_layers=12, dim=512, num_heads=8, window=64, ssm_state=32)
    T = 8192
    high_f3 = block_flops_per_token_decode(cfg, (0.0, 0.5, 0.2, 0.3), T)
    low_f3 = block_flops_per_token_decode(cfg, (0.1, 0.6, 0.2, 0.1), T)
    assert low_f3["total"] < high_f3["total"]


def test_fractions_sum_check_not_required():
    """Profiling accepts any nonneg fractions; sum != 1 just rescales."""
    cfg = ChimeraConfig(num_layers=4, dim=64, num_heads=4)
    out = block_flops_per_token_decode(cfg, (0.0, 1.0, 0.0, 0.0), 64)
    assert out["full"] == 0.0
    assert out["ssm"] > 0
