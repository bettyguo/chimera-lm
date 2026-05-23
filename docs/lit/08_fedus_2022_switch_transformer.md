# Fedus et al. 2022 — Switch Transformer

> **Status:** TODO — unread.

## Primitive operation
TODO — top-1 routing across N FFN experts per token, with an auxiliary
load-balance loss `K · Σ f_k · P_k`.

## Complexity
TODO — sparse FFN; per-token compute drops as #active experts / #total experts.

## Failure mode the paper itself admits
TODO — load imbalance under hard routing without the aux loss; expert
collapse; training instability requiring capacity factor and z-loss.

## Empirically strongest result
TODO

## Relevance to CHIMERA
**The aux-loss baseline our `AuxFreeBalancer` replaces.** `chimera/losses.py::
load_balance_aux_loss` implements the Switch-style `K · Σ f_k · P_k` for use
as an ablation. Default `aux_loss_coef = 0` (we use aux-loss-free instead);
toggle on to compare.

Switch's *capacity factor* (Phase 3) and deterministic-downgrade strategy
also come from this paper.

## Cited by
- `chimera/losses.py::load_balance_aux_loss`
- `docs/routing.md` (capacity factor discussion)
