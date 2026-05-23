"""Router + aux-free balancer tests."""

import torch

from chimera.modules.router import NUM_MODES, AuxFreeBalancer, MixerRouter


def test_router_shapes_soft():
    torch.manual_seed(0)
    router = MixerRouter(dim=32)
    x = torch.randn(2, 8, 32)
    out = router(x, mode="soft")
    assert out.weights.shape == (2, 8, NUM_MODES)
    assert torch.allclose(out.weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)
    assert out.hard_index.shape == (2, 8)
    assert out.one_hot.shape == (2, 8, NUM_MODES)


def test_router_shapes_hard():
    torch.manual_seed(0)
    router = MixerRouter(dim=32)
    x = torch.randn(2, 8, 32)
    out = router(x, mode="hard_top1")
    # Hard one-hot must equal the argmax in forward.
    rounded = out.one_hot.round()
    assert rounded.sum(dim=-1).eq(1.0).all()
    # Straight-through: backward should flow to router params via the soft path.
    out.one_hot.sum().backward()
    assert router.proj.weight.grad is not None
    assert router.proj.weight.grad.abs().sum().item() > 0


def test_router_hard_one_hot_matches_argmax():
    torch.manual_seed(0)
    router = MixerRouter(dim=16)
    x = torch.randn(1, 4, 16)
    out = router(x, mode="hard_top1")
    picked = out.one_hot.argmax(dim=-1)
    assert torch.equal(picked, out.hard_index)


def test_balancer_pushes_toward_target():
    """If the observed distribution stays at all-mode-1, the bias for mode 1
    should fall negative and others should rise."""
    torch.manual_seed(0)
    bal = AuxFreeBalancer(target=(0.10, 0.60, 0.20, 0.10), bias_step=0.1)
    # Observe 1000 batches that are all mode-1.
    hard = torch.full((4, 32), 1, dtype=torch.long)
    for _ in range(200):
        bal.observe_and_update(hard)
    # mode-1 bias should be negative; others non-negative.
    # (bias is registered as a buffer; stubs return Module — type-ignored for index access)
    assert bal.bias[1] < 0  # type: ignore[index]
    assert (bal.bias[[0, 2, 3]] > 0).all()  # type: ignore[index]


def test_balancer_clamp():
    bal = AuxFreeBalancer(target=(0.25, 0.25, 0.25, 0.25), bias_step=1.0, bias_clip=2.0)
    # Force observation to one mode only, very many steps.
    hard = torch.zeros(4, 8, dtype=torch.long)
    for _ in range(1000):
        bal.observe_and_update(hard)
    assert bal.bias.abs().max().item() <= 2.0 + 1e-6


def test_balancer_no_update_in_eval():
    """Router with update_balancer=False must not mutate the balancer."""
    torch.manual_seed(0)
    router = MixerRouter(dim=16)
    router.eval()
    initial = router.balancer.bias.clone()
    x = torch.randn(2, 8, 16)
    router(x, mode="hard_top1", update_balancer=True)  # eval mode, no update
    assert torch.equal(router.balancer.bias, initial)
