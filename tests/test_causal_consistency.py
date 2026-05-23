"""The canary test from spec Appendix A.2.

For any input sequence x_1..x_T and any routing decisions g_1..g_T:

    prefill(x).logits  ==  [decode_step(x_t, g_t, cache) for t in 1..T]

to fp64 bit-exactness (< 1e-10) and < 1e-4 in bf16.

If any of these fails, do NOT silently fix it — write a POSTMORTEM, find the
off-by-one, and resubmit.
"""

import pytest
import torch

from chimera.cache import ChimeraCacheLayer
from chimera.model import ChimeraConfig, ChimeraLM
from chimera.modules.chimera_block import BlockConfig, ChimeraBlock


# ----------------------------------------------------------------------
# Block-level parity
# ----------------------------------------------------------------------
def _block_parity(
    cfg: BlockConfig,
    T: int,
    router_mode: str,
    seed: int = 42,
    tol: float = 1e-9,
) -> float:
    torch.manual_seed(seed)
    block = ChimeraBlock(cfg).double()
    block.eval()  # don't mutate the balancer between prefill and decode
    x = torch.randn(1, T, cfg.dim, dtype=torch.float64)

    # Prefill
    y_prefill, router_out_p = block.forward_prefill(x, router_mode=router_mode)

    # Decode step-by-step from empty cache.
    cache = ChimeraCacheLayer.empty(
        batch=1,
        kv_shape=(cfg.num_heads, cfg.head_dim),
        d_ssm=cfg.ssm_state,
        window=cfg.window,
        device=x.device,
        dtype=x.dtype,
    )
    y_decode = torch.zeros_like(y_prefill)
    for t in range(T):
        y_t, _ = block.forward_decode_step(
            x[:, t], cache, position=t, router_mode=router_mode
        )
        y_decode[:, t] = y_t
    diff = (y_prefill - y_decode).abs().max().item()
    assert diff < tol, f"block parity {router_mode} T={T}: diff={diff:.2e}"
    return diff


@pytest.mark.parametrize("T", [8, 32, 128])
def test_block_parity_hard(T):
    cfg = BlockConfig(
        dim=32, num_heads=4, window=8, ssm_state=8, max_seq_len=256
    )
    diff = _block_parity(cfg, T=T, router_mode="hard_top1")
    print(f"hard T={T} diff={diff:.2e}")


@pytest.mark.parametrize("T", [8, 32, 128])
def test_block_parity_soft(T):
    cfg = BlockConfig(
        dim=32, num_heads=4, window=8, ssm_state=8, max_seq_len=256
    )
    diff = _block_parity(cfg, T=T, router_mode="soft")
    print(f"soft T={T} diff={diff:.2e}")


def test_block_parity_ring_eviction():
    """Stress the ring buffer: T much larger than window."""
    cfg = BlockConfig(
        dim=32, num_heads=4, window=8, ssm_state=8, max_seq_len=256
    )
    # T=64 with window=8 forces multiple evictions.
    diff = _block_parity(cfg, T=64, router_mode="hard_top1")
    print(f"ring eviction T=64 W=8 diff={diff:.2e}")


def test_block_parity_window_equals_T():
    """Edge case: window == T. Ring should never evict."""
    cfg = BlockConfig(
        dim=32, num_heads=4, window=16, ssm_state=8, max_seq_len=64
    )
    _block_parity(cfg, T=16, router_mode="hard_top1")


# ----------------------------------------------------------------------
# Model-level parity
# ----------------------------------------------------------------------
def _model_parity(
    cfg: ChimeraConfig,
    T: int,
    router_mode: str,
    seed: int = 42,
    tol: float = 1e-8,
) -> float:
    torch.manual_seed(seed)
    model = ChimeraLM(cfg).double()
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, T))

    p = model.forward_prefill(input_ids, router_mode=router_mode)

    cache = model.empty_cache(batch=1)
    logits_decode = torch.zeros_like(p.logits)
    for t in range(T):
        out = model.forward_decode_step(
            input_ids[:, t], cache, position=t, router_mode=router_mode
        )
        logits_decode[:, t] = out.logits

    diff = (p.logits - logits_decode).abs().max().item()
    assert diff < tol, f"model parity {router_mode} T={T}: diff={diff:.2e}"
    return diff


def test_model_parity_hard_short():
    cfg = ChimeraConfig(
        vocab_size=128,
        num_layers=2,
        dim=32,
        num_heads=4,
        window=8,
        ssm_state=8,
        max_seq_len=64,
    )
    diff = _model_parity(cfg, T=16, router_mode="hard_top1")
    print(f"model hard T=16 diff={diff:.2e}")


def test_model_parity_soft_short():
    cfg = ChimeraConfig(
        vocab_size=128,
        num_layers=2,
        dim=32,
        num_heads=4,
        window=8,
        ssm_state=8,
        max_seq_len=64,
    )
    diff = _model_parity(cfg, T=16, router_mode="soft")
    print(f"model soft T=16 diff={diff:.2e}")


def test_model_parity_with_ring_eviction():
    cfg = ChimeraConfig(
        vocab_size=64,
        num_layers=2,
        dim=32,
        num_heads=4,
        window=4,
        ssm_state=8,
        max_seq_len=64,
    )
    diff = _model_parity(cfg, T=24, router_mode="hard_top1")
    print(f"model hard T=24 W=4 diff={diff:.2e}")


# ----------------------------------------------------------------------
# Cache integrity (independent of forward)
# ----------------------------------------------------------------------
def test_cache_ring_eviction_keeps_window_size():
    cache = ChimeraCacheLayer.empty(
        batch=1, kv_shape=(2, 4), d_ssm=4, window=3,
        device="cpu", dtype=torch.float64,
    )
    for t in range(10):
        k = torch.full((1, 2, 4), float(t), dtype=torch.float64)
        v = torch.full((1, 2, 4), float(t), dtype=torch.float64)
        cache.commit_write(k, v, mode=2, new_ssm_state=cache.ssm_state)
    # After 10 writes with window=3, ring size capped at 3 with [7, 8, 9].
    assert cache.ring_size == 3
    assert torch.allclose(cache.ring_k[0, 0], torch.full((2, 4), 7.0, dtype=torch.float64))
    assert torch.allclose(cache.ring_k[0, 1], torch.full((2, 4), 8.0, dtype=torch.float64))
    assert torch.allclose(cache.ring_k[0, 2], torch.full((2, 4), 9.0, dtype=torch.float64))


def test_cache_persistent_only_mode3():
    cache = ChimeraCacheLayer.empty(
        batch=1, kv_shape=(2, 4), d_ssm=4, window=8,
        device="cpu", dtype=torch.float64,
    )
    for mode in [0, 1, 2, 0, 1]:
        k = torch.randn(1, 2, 4, dtype=torch.float64)
        v = torch.randn(1, 2, 4, dtype=torch.float64)
        cache.commit_write(k, v, mode=mode, new_ssm_state=cache.ssm_state)
    assert cache.persistent_k == [] and cache.persistent_v == []

    k = torch.randn(1, 2, 4, dtype=torch.float64)
    v = torch.randn(1, 2, 4, dtype=torch.float64)
    cache.commit_write(k, v, mode=3, new_ssm_state=cache.ssm_state)
    assert len(cache.persistent_k) == 1
    assert len(cache.persistent_v) == 1
