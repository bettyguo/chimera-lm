"""Shape tests for ChimeraBlock."""

import torch

from chimera.modules.chimera_block import BlockConfig, ChimeraBlock
from chimera.modules.router import NUM_MODES


def test_block_forward_shape_soft():
    torch.manual_seed(0)
    cfg = BlockConfig(dim=64, num_heads=4, window=8, ssm_state=16, max_seq_len=64)
    block = ChimeraBlock(cfg)
    block.eval()
    x = torch.randn(2, 32, 64)
    y, router_out = block.forward_prefill(x, router_mode="soft")
    assert y.shape == x.shape
    assert router_out.weights.shape == (2, 32, NUM_MODES)


def test_block_forward_shape_hard():
    torch.manual_seed(0)
    cfg = BlockConfig(dim=64, num_heads=4, window=8, ssm_state=16, max_seq_len=64)
    block = ChimeraBlock(cfg)
    block.eval()
    x = torch.randn(2, 32, 64)
    y, router_out = block.forward_prefill(x, router_mode="hard_top1")
    assert y.shape == x.shape


def test_block_gradient_flow_soft():
    """Soft routing must let gradient flow through ALL four mode paths."""
    torch.manual_seed(0)
    cfg = BlockConfig(dim=32, num_heads=2, window=4, ssm_state=8, max_seq_len=32)
    block = ChimeraBlock(cfg)
    # Note: train mode would update the balancer; we want to test gradient flow,
    # which doesn't depend on the balancer update — use eval to keep balancer fixed.
    block.eval()
    x = torch.randn(2, 16, 32, requires_grad=True)
    y, _ = block.forward_prefill(x, router_mode="soft")
    y.sum().backward()

    # Router params should receive gradient (from soft weights).
    assert block.router.proj.weight.grad is not None
    assert block.router.proj.weight.grad.abs().sum().item() > 0
    # SSM params should receive gradient (mode 1 is non-trivial).
    assert block.ssm.in_proj.weight.grad is not None
    assert block.ssm.in_proj.weight.grad.abs().sum().item() > 0
    # Attention params should receive gradient (modes 2 and 3 use them).
    assert block.proj_q.weight.grad is not None
    assert block.proj_q.weight.grad.abs().sum().item() > 0
    # FFN params should receive gradient.
    assert block.ffn.w_down.weight.grad is not None
    assert block.ffn.w_down.weight.grad.abs().sum().item() > 0
