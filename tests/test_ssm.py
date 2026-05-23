"""Toy SSM interface tests — step / forward_prefill equivalence."""

import torch

from chimera.modules.ssm import ToySSM


def test_step_matches_prefill():
    """forward_prefill must equal the sequential application of step()."""
    torch.manual_seed(0)
    B, T, D, S = 2, 16, 8, 4
    ssm = ToySSM(dim=D, state_size=S).double()
    x = torch.randn(B, T, D, dtype=torch.float64)
    y_prefill, final_state = ssm.forward_prefill(x)

    # Sequential
    state = ssm.empty_state(B, device=x.device, dtype=x.dtype)
    outs = []
    for t in range(T):
        y_t, state = ssm.step(x[:, t], state)
        outs.append(y_t)
    y_seq = torch.stack(outs, dim=1)
    assert torch.allclose(y_prefill, y_seq, atol=1e-12)
    assert torch.allclose(final_state, state, atol=1e-12)


def test_state_evolves_with_input():
    """An input perturbation should propagate through future states."""
    torch.manual_seed(0)
    ssm = ToySSM(dim=4, state_size=3).double()
    x = torch.zeros(1, 5, 4, dtype=torch.float64)
    y0, _ = ssm.forward_prefill(x)
    x[:, 0, 0] = 1.0
    y1, _ = ssm.forward_prefill(x)
    # Position 0 should differ.
    assert not torch.allclose(y0[:, 0], y1[:, 0], atol=1e-6)
    # And the diff should propagate forward (recurrence).
    assert not torch.allclose(y0[:, -1], y1[:, -1], atol=1e-6)
