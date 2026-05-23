# Routing and load balancing

## The router

`MixerRouter` (in `chimera/modules/router.py`) emits a distribution over `K=4`
modes per (token, layer):

```
h    = LayerNorm(x)                                    (B, T, D)
ℓ    = h W_router + b_aux                              (B, T, K)
g    = softmax(ℓ / τ, dim=-1)                          (B, T, K)
m̂   = ℓ.argmax(dim=-1)                                (B, T)
```

`b_aux` is a *non-gradient* bias maintained by the aux-loss-free balancer.

Three router modes:

- **`"soft"`** — return full distribution `g`. Caller computes
  `y = Σ_k g_k * mixer_k(x)`. Used during early training.
- **`"hard_top1"`** — return one-hot of `m̂` with a straight-through gradient
  surrogate. Caller computes only `mixer_{m̂}(x)` (or all four with a one-hot
  mix, as we do during prefill for code simplicity). Deployment target.
- **`"gumbel"`** — sample with Gumbel-softmax + straight-through. Available
  for ablations; not the default.

Straight-through hard one-hot: forward uses `one_hot(m̂)`, backward uses the
soft `g`. This keeps gradient flowing to `W_router`.

## Aux-loss-free balancing (spec A.1)

We want the routing fractions to converge to a *target* distribution `f*`
(default `(0.10, 0.60, 0.20, 0.10)` for identity / SSM / SWA / full). Naive
softmax routing collapses to mode 1 (cheapest, cleanest gradient). DeepSeek-V3's
solution: don't add an auxiliary loss; add a per-mode bias adjusted by an
online controller.

```
f̄_k ← β f̄_k + (1 − β) f_k_observed               # EMA, β = 0.95
b_k ← clip(b_k + γ · sign(f_k* − f̄_k), −b_max, +b_max)   # γ = 1e-3, b_max = 4
```

The bias is added to the router logits before softmax; the controller updates
`b_k` based on how the empirical fraction `f̄_k` compares to the target. **No
gradients flow through `b_k`** — the bias is a pure control-system mechanism,
not part of the loss landscape.

Why this works:

- A 1.0 bias differential ≈ 2.7× post-softmax probability shift; the clamp at
  ±4 corresponds to ~55× swing — far more than typically needed.
- With γ = 1e-3, a 50% under-utilization for 1000 steps accumulates a 1.0
  bias advantage. Fast enough to escape collapse, slow enough to not oscillate.
- The EMA filters per-batch noise; without it, the bias chases batch-level
  fluctuations and oscillates.
- **Per-layer** biases (not shared) — different layers will converge to
  different distributions (we expect early layers SSM-heavy, late layers
  attention-heavier on rare tokens).

## Auxiliary loss (off by default)

`chimera/losses.py` provides a Switch-Transformer-style `load_balance_aux_loss`
for *ablation only*. It's `K · Σ_k f_k · P_k` where `f_k` is hard fraction and
`P_k` is soft probability. The default coefficient is 0 (aux-loss-free).

## Routing diagnostics (logged but not differentiated)

`routing_diagnostics` returns:

- **Per-layer hard fractions** `(K,)` — how often each mode actually fired.
- **Per-layer soft mass** `(K,)` — mean of the soft weights.
- **Per-layer entropy** — scalar; how peaked the soft distribution is.

The trainer must log all three every step. The spec says: "alert if any mode
drops below 1%". That's a P0 anomaly; usually means the balancer is broken or
the controller constants are wrong for the data.

## Decode-time routing

At decode, the router runs the same way per step. The hard argmax determines
which writes are committed:

- SSM state: always updated (mode-1's read uses the post-update state).
- Ring buffer: always written (Resolution A).
- Persistent KV: appended only when `m̂ == 3`.

The balancer **does not update** at decode (`update_balancer=False` or
`model.eval()`). Decode-time routing distributions should be logged for
inspection but never fed back to the controller.

## Capacity factor and overflow (planned, Phase 3)

`MixerRouter` does *not* enforce capacity yet. The spec specifies:

```
C_k = ⌈capacity_factor · f_k* · T_batch⌉   capacity_factor = 1.0
```

with **deterministic downgrade** for overflow: a token wanting mode-3 when
mode-3 is full gets demoted to mode 2, then mode 1, then mode 0. The
aux-loss-free controller handles long-term balance; capacity factor + downgrade
handles short-term spikes.

Phase 3 will add this as a wrapper around `MixerRouter` that runs at the batch
level. We don't add it in Phase 1 because (a) it's a training-time concern and
(b) per-token soft routing during warmup makes capacity factor moot.
