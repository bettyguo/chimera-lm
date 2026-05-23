# DeepSeek-AI 2024 — DeepSeek-V3

> **Status:** TODO — unread.

## Primitive operation
TODO — fine-grained MoE FFN with **auxiliary-loss-free** load balancing: a
non-gradient bias added to router logits, updated by a deterministic
controller toward a target distribution.

## Complexity
TODO

## Failure mode the paper itself admits
TODO — extreme imbalance can still occur under sudden distribution shifts;
the controller has finite response speed.

## Empirically strongest result
TODO — headline DeepSeek-V3 perplexity / benchmarks. The relevant ablation
shows aux-loss-free *outperforms* aux-loss at matched compute (because the
LM loss isn't competing with a routing penalty).

## Relevance to CHIMERA
**Source of the balancer math.** Our `AuxFreeBalancer` in
`chimera/modules/router.py` is a direct port of DeepSeek-V3's controller:

```
f̄_k ← β f̄_k + (1−β) f_k_observed
b_k ← clip(b_k + γ · sign(f_k* − f̄_k), -b_max, +b_max)
```

with `β = 0.95, γ = 1e-3, b_max = 4.0`. See spec Appendix A.1 and
`docs/routing.md` for our notes on tuning constants.

The smoke run in `scripts/smoke_train_nano.py` exposed a layer-1 collapse
at 200 steps — see `POSTMORTEM.md` 2026-05-23. This is expected at short
horizon; production runs at 4B tokens give the controller time.

## Cited by
- `chimera/modules/router.py::AuxFreeBalancer`
- `docs/routing.md`
- `POSTMORTEM.md` (2026-05-23 entry)
