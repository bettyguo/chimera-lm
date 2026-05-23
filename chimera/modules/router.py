"""Mixer router with auxiliary-loss-free load balancing (spec A.1).

The router emits a distribution over K mixing modes per (token, layer). We
support three modes of operation:

  - SOFT       — return full softmax distribution; caller computes
                 y = sum_k g_k * mixer_k(x). Used during early training.
  - HARD_TOP1  — return one-hot top-1 with straight-through gradient.
                 Used after warmup, when one mode per token is computed.
  - GUMBEL     — sample one mode via Gumbel-softmax; straight-through.
                 Optional ablation; not used by default.

Aux-loss-free balancing (DeepSeek-V3): maintain a non-gradient bias `b_k` per
mode that is added to the router logits. The bias is updated by a deterministic
sign-controller toward a target distribution. No second loss term, so the LM
objective stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mode identifiers — must match cache.commit_write / decode dispatch.
MODE_IDENTITY = 0
MODE_SSM = 1
MODE_SWA = 2
MODE_FULL = 3
NUM_MODES = 4


@dataclass
class RouterOutput:
    """Router emission for a single forward pass.

    Attributes:
      weights:        (B, T, K) — soft routing distribution (post-bias softmax).
                      Sums to 1 over the K axis.
      hard_index:     (B, T) long — argmax mode (computed regardless of mode).
      one_hot:        (B, T, K) — straight-through one-hot of hard_index in
                      HARD_TOP1/GUMBEL modes, else the same as `weights`.
      logits:         (B, T, K) — pre-softmax (gradient-bearing logits +
                      detached bias). Useful for logging.
    """

    weights: torch.Tensor
    hard_index: torch.Tensor
    one_hot: torch.Tensor
    logits: torch.Tensor


class AuxFreeBalancer(nn.Module):
    """Non-gradient bias controller that pushes routing fractions toward target.

    Per spec A.1:
        f̄_k ← β f̄_k + (1−β) f_k_observed     # EMA
        b_k ← clip(b_k + γ sign(f_k* − f̄_k), -b_max, +b_max)

    The bias is registered as a buffer (no gradient). The EMA fractions are
    also buffers so they survive `.state_dict()`.

    The controller has a per-layer instance — different layers will converge to
    different distributions, and a shared controller forces them into the same
    regime (spec A.1, "Per-layer or shared?").
    """

    def __init__(
        self,
        num_modes: int = NUM_MODES,
        target: tuple[float, ...] = (0.10, 0.60, 0.20, 0.10),
        ema_beta: float = 0.95,
        bias_step: float = 1e-3,
        bias_clip: float = 4.0,
    ) -> None:
        super().__init__()
        if len(target) != num_modes:
            raise ValueError(f"target has {len(target)} modes, expected {num_modes}")
        if not abs(sum(target) - 1.0) < 1e-6:
            raise ValueError(f"target must sum to 1, got {sum(target)}")
        self.num_modes = num_modes
        self.ema_beta = ema_beta
        self.bias_step = bias_step
        self.bias_clip = bias_clip
        self.register_buffer("target", torch.tensor(target, dtype=torch.float32))
        self.register_buffer("ema_fraction", torch.tensor(target, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(num_modes, dtype=torch.float32))

    @torch.no_grad()
    def observe_and_update(self, hard_index: torch.Tensor) -> torch.Tensor:
        """Update EMA + bias from observed routing decisions.

        hard_index: (B, T) long. Returns the observed per-batch fraction (K,)
        for logging.
        """
        flat = hard_index.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_modes).to(torch.float32)
        fraction = counts / max(flat.numel(), 1)
        # EMA update.
        self.ema_fraction.mul_(self.ema_beta).add_(fraction, alpha=1.0 - self.ema_beta)
        # Sign-step bias update.
        diff = self.target - self.ema_fraction
        self.bias.add_(self.bias_step * torch.sign(diff))
        self.bias.clamp_(-self.bias_clip, self.bias_clip)
        return fraction


class MixerRouter(nn.Module):
    """Linear router with LN-on-input + aux-free balancer.

    Forward pass:
        h = LayerNorm(x)               # (B, T, D)
        ℓ = h W_router + b_aux         # (B, T, K) — b_aux is non-gradient
        weights = softmax(ℓ / τ, dim=-1)
        hard_index = ℓ.argmax(dim=-1)
        one_hot = straight_through(hard_index, weights)

    Notes:
      - The router consumes the *current token's* representation (the
        prefill input, never post-block output) to avoid future-leak.
      - τ (temperature) defaults to 1.0; the aux-free balancer handles
        long-term distribution shape, so we don't anneal τ.
    """

    def __init__(
        self,
        dim: int,
        num_modes: int = NUM_MODES,
        target: tuple[float, ...] = (0.10, 0.60, 0.20, 0.10),
        temperature: float = 1.0,
        balancer: AuxFreeBalancer | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_modes = num_modes
        self.temperature = temperature
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, num_modes, bias=False)
        self.balancer = balancer if balancer is not None else AuxFreeBalancer(
            num_modes=num_modes, target=target
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mode: str = "soft",
        update_balancer: bool = True,
    ) -> RouterOutput:
        """Route per-token.

        x:    (B, T, D)
        mode: "soft" | "hard_top1" | "gumbel"
        update_balancer: when True (training), call observe_and_update on the
                        hard argmax. Set False at eval/decode.
        """
        h = self.norm(x)
        # Bias is detached / non-gradient — its update path is in observe_and_update.
        bias = self.balancer.bias.to(h.dtype).detach()
        logits = self.proj(h) + bias  # (B, T, K)
        scaled = logits / self.temperature
        weights = F.softmax(scaled, dim=-1)
        hard_index = scaled.argmax(dim=-1)

        if mode == "soft":
            one_hot = weights
        elif mode == "hard_top1":
            one_hot_hard = F.one_hot(hard_index, num_classes=self.num_modes).to(weights.dtype)
            # Straight-through: forward uses hard one-hot, backward uses soft weights.
            one_hot = one_hot_hard + (weights - weights.detach())
        elif mode == "gumbel":
            # Sample with Gumbel noise; straight-through to one-hot.
            gumbel = -torch.log(-torch.log(torch.rand_like(scaled).clamp_min(1e-9)).clamp_min(1e-9))
            noisy = (scaled + gumbel) / self.temperature
            sampled_idx = noisy.argmax(dim=-1)
            one_hot_hard = F.one_hot(sampled_idx, num_classes=self.num_modes).to(weights.dtype)
            soft = F.softmax(noisy, dim=-1)
            one_hot = one_hot_hard + (soft - soft.detach())
            hard_index = sampled_idx
        else:
            raise ValueError(f"unknown router mode {mode!r}")

        if update_balancer and self.training:
            self.balancer.observe_and_update(hard_index)

        return RouterOutput(
            weights=weights,
            hard_index=hard_index,
            one_hot=one_hot,
            logits=logits,
        )
