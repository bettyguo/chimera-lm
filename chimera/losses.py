"""Loss computations and routing diagnostics.

LM loss is plain cross-entropy. The router uses *aux-loss-free* balancing
(spec A.1), so we do NOT add a routing loss term — the bias controller does
the balancing. We do compute and report:

  - per-layer routing fractions (which mode fired how often)
  - mode-mass over the batch (mass of soft weights per mode)
  - routing entropy (per token, per layer) — useful as a regularization
    diagnostic, not a loss term

If you want classical Switch-style aux-loss as an ablation, use
`load_balance_aux_loss` and add it to the LM loss with a small coefficient.
This is intentionally OFF by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chimera.modules.router import RouterOutput


@dataclass
class TrainStepLoss:
    """Container for the bits a training loop wants to log."""

    loss: torch.Tensor  # scalar — the loss we backprop
    ce: torch.Tensor    # scalar — plain cross-entropy
    aux: torch.Tensor   # scalar — auxiliary load-balance term (zero by default)
    per_layer_fractions: list[torch.Tensor]   # each: (K,) — empirical f_k
    per_layer_mass: list[torch.Tensor]        # each: (K,) — mean of soft weights
    per_layer_entropy: list[torch.Tensor]     # each: scalar — mean entropy


def cross_entropy_lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int = -100
) -> torch.Tensor:
    """Standard next-token-prediction CE.

    logits:  (B, T, V) — prediction for position t (target is x_{t+1})
    targets: (B, T) long — already shifted by the caller, or padded with ignore_index
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), ignore_index=ignore_index
    )


def load_balance_aux_loss(
    router_outputs: list[RouterOutput], target: torch.Tensor | None = None
) -> torch.Tensor:
    """Switch-Transformer-style auxiliary load-balance loss.

    Defined as K * sum_k f_k * P_k where:
      f_k = fraction of tokens routed to mode k (hard, non-differentiable)
      P_k = mean of soft probability assigned to mode k (differentiable)

    Lower is better when both f and P are uniform. With aux-loss-free balancing
    enabled (the default), this term should NOT be added to the loss. Useful
    only as an ablation toggle.

    Returns: scalar.
    """
    if not router_outputs:
        return torch.tensor(0.0)
    total = torch.zeros((), device=router_outputs[0].weights.device)
    K = router_outputs[0].weights.shape[-1]
    for r in router_outputs:
        weights = r.weights  # (B, T, K) — soft
        hard = F.one_hot(r.hard_index, num_classes=K).to(weights.dtype)  # (B, T, K)
        # Per-batch fraction (non-diff w.r.t. router weights via the argmax)
        f_k = hard.mean(dim=(0, 1))  # (K,)
        P_k = weights.mean(dim=(0, 1))  # (K,)
        # K * sum f * P — uniform => K * K * (1/K)*(1/K) = 1
        total = total + K * (f_k * P_k).sum()
    return total / len(router_outputs)


def routing_diagnostics(router_outputs: list[RouterOutput]) -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    """Compute per-layer fractions, soft mass, and entropy for logging.

    Returns three parallel lists (one entry per layer):
      fractions:  (K,) — empirical hard-routing fraction (non-diff)
      mass:       (K,) — mean soft weight (diff but typically logged not used)
      entropy:    scalar — mean entropy of the soft routing distribution
    """
    fracs, masses, entropies = [], [], []
    for r in router_outputs:
        K = r.weights.shape[-1]
        hard = F.one_hot(r.hard_index, num_classes=K).to(r.weights.dtype)
        fracs.append(hard.mean(dim=(0, 1)).detach())
        masses.append(r.weights.mean(dim=(0, 1)).detach())
        ent = -(r.weights * torch.log(r.weights.clamp_min(1e-12))).sum(dim=-1).mean()
        entropies.append(ent.detach())
    return fracs, masses, entropies


def chimera_train_step_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    router_outputs: list[RouterOutput],
    *,
    ignore_index: int = -100,
    aux_loss_coef: float = 0.0,
) -> TrainStepLoss:
    """One-stop loss for a training step.

    By default `aux_loss_coef = 0` (aux-loss-free). Set positive to enable the
    Switch-style aux term for an ablation.
    """
    ce = cross_entropy_lm_loss(logits, targets, ignore_index=ignore_index)
    aux = (
        load_balance_aux_loss(router_outputs)
        if aux_loss_coef > 0
        else torch.zeros((), device=logits.device)
    )
    fracs, masses, entropies = routing_diagnostics(router_outputs)
    return TrainStepLoss(
        loss=ce + aux_loss_coef * aux,
        ce=ce.detach(),
        aux=aux.detach(),
        per_layer_fractions=fracs,
        per_layer_mass=masses,
        per_layer_entropy=entropies,
    )
