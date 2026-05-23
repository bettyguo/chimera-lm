# POSTMORTEM log

Per spec §0 operating loop: anomalies during training get documented here.
*Never silently fix anomalies — write down what surprised you and what you
changed.*

---

## 2026-05-23 — Layer-1 routing collapse in nano smoke run

### Observation

`scripts/smoke_train_nano.py` (200 steps, 2-layer 64-dim model on synthetic
MQAR with N=6 kv pairs, M=3 queries) showed:

- Loss decreased: **63.87 → 2.32** (validation 60.89 → 2.24). Training works.
- Layer 0 routing converged near target: `[0.04, 0.59, 0.18, 0.19]` against
  target `(0.10, 0.60, 0.20, 0.10)`. Healthy.
- **Layer 1 collapsed to mode-1 (SSM)**: `[0.00, 0.96, 0.02, 0.02]`. Mode 0
  hit 0%. This is the spec's named "router collapses to all-mode-1" failure
  mode (Risk Table row 1).

### Why it happened (analysis)

The aux-loss-free controller uses `γ = 1e-3` and `sign()` updates. In 200 steps
the *maximum* accumulated |Δbias| is 0.2. The LM-loss gradient pushed Layer 1
toward mode-1 faster than the balancer could push back. By the time the bias
became meaningful (~step 1000+), the model had already specialized.

Layer 0 did *not* collapse — presumably because Layer 0 sees raw embeddings
and the routing signal is stronger there, so it stabilized before the balancer
needed to intervene.

### What this confirms

- The collapse mode is real, just as the spec warns.
- The default `γ = 1e-3` is appropriate for the *4B-token nano training run*
  the spec specifies. At 200 steps it's far too small.
- Loss does decrease and the trainer machinery is sound — the issue is the
  *routing dynamics*, not the trainer or model.

### What we DID NOT change

We did **not** silently bump γ in the default config. The spec's constants are
chosen for production-scale runs; changing them at toy scale would mask the
same failure at production scale.

### What to do at production scale

Per spec A.1:

- If you see `b_k` saw-tooth oscillating at >> 20-step frequency, increase β
  (to 0.99) or decrease γ (to 3e-4).
- If you see slow drift away from `f^*`, the opposite.

For the smoke-run scale specifically, an investigator might experiment with
`γ = 1e-2` and `bias_step` set per-call. We resist baking this into defaults
because the production run won't need it.

### What to do at toy scale (Phase 2 investigation)

If we want a clean smoke run that converges in 200 steps without collapse:

1. Bump `AuxFreeBalancer(bias_step=1e-2)` (10× faster controller).
2. Increase the number of layers from 2 to 4+ — the collapse risk is per-layer,
   and with more layers, on average more will stay diverse.
3. Use a less SSM-favoring data distribution (MQAR-only is biased; mix with
   selective-copy or with WikiText-style next-token).
4. Initialize the router slightly biased toward modes 2/3 (the spec
   recommends against this but explicitly notes it as a corrective lever).

None of these are baked into the smoke driver. If a future implementer wants
to experiment, do it in a new script and write a new POSTMORTEM entry.

### Tests added in response

- `tests/test_train.py::test_balancer_responds_to_observation` — verifies the
  controller is wired in and actually moves bias values. Catches the case
  where someone accidentally disconnects the controller (the silent failure).
- The smoke driver itself emits a `WARNING: collapsed modes` line if any mode
  drops below 1%. Visible signal.

### Follow-up owners / dates

- Re-run on GPU with Mamba-2 SSD + flash-attn, ≥4 layers, ≥1k steps. Expected:
  the controller has enough time to push Layer 1 back toward target. If not,
  open another postmortem.

---
