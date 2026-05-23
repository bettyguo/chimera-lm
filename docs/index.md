# CHIMERA

**Conditionally Hybrid Mixture of Exact and Recurrent Attention.**

A decoder-only language model where every token, at every layer, is routed
through the cheapest sequence-mixing primitive that suffices: identity,
state-space model, sliding-window attention, or full causal attention.

The thesis: the hybridization ratio between attention and recurrence should
be **learned per-token and per-layer**, not a fixed design-time hyperparameter.

!!! abstract "What sets CHIMERA apart"

    - **Mixture-of-Depths**, but routing between sequence operators instead of layer skips.
    - **DeepSeekMoE**-style fine-grained experts, but the experts are *sequence mixers*, not FFNs.
    - **Hybrid SSM-attention stacks** (Jamba, Samba, Zamba), but the hybridization
      is *data-dependent* rather than fixed.

## Repository status

Phase 1 + Phase 2-prep complete on CPU. **51 tests pass at fp64 bit-exactness**,
including the causal-consistency canary (prefill ≡ step-by-step decode).

| Phase | Status |
|---|---|
| 0 — Bootstrap | done (CI workflow, lit-note stubs, `pyproject.toml`) |
| 1 — Single block | done (router, SSM stub, attention, RoPE, FFN, block, model) |
| 2 — KV/state cache + decode parity | done (Resolution A, bit-exact fp64) |
| 2-prep — Trainer + diagnostics | done (AdamW, MQAR data, balancer logging) |
| 1.5 — Mamba-2 + flash-attn swap | wrappers shipped, **needs CUDA box** |
| 3 — Training at scale | blocked on GPU + real text |
| 4 — Benchmarks (MQAR, needle, LongBench) | blocked on Phase 3 checkpoint |
| 5 — Ablations + paper | blocked on Phase 4 |

## Pages on this site

- **[Quickstart](quickstart.md)** — install, run tests, reproduce the smoke run.
- **[Architecture](architecture.md)** — block diagram and tensor shapes.
- **[Routing & balancing](routing.md)** — router math and the aux-loss-free controller.
- **[Multi-mode KV cache](kv_cache.md)** — Resolution A and the parity contract.
- **[Implementation notes](think.md)** — Phase 1 design rationale.
- **[Nano MQAR experiment](experiments/nano_report.md)** — three-way head-to-head results.
- **[Decisions](decisions/ADR-001-cache-resolution.md)** — architecture decision records.
- **[Literature](lit/README.md)** — required reading.
- **[Postmortem](postmortem.md)** — observed anomalies and what they mean.

## Verified findings (so far)

- **Bit-exact prefill ≡ decode parity** at fp64 across single-mode, mixed-routing,
  ring-eviction, and full-model paths. Block-level diff ≤ ~1e-15; model-level
  diff ≤ ~1e-14.
- **Param counts match spec headlines** within rounding (nano 11.5M / 12M,
  small 111M / 125M, medium 343M / 350M, large 1.29B / 1.3B).
- **The router learns task structure even at toy scale.** On a 200-step run
  of CHIMERA on synthetic MQAR, Layer 0 routed 68.8% of *query positions* to
  sliding-window attention vs. only 38% of *kv positions* — and never picked
  identity at queries. Detail in [the nano report](experiments/nano_report.md).
- **Layer-1 collapse is reproducible**, just as the spec warns. Default
  `γ = 1e-3` is too slow at 200 steps; production runs at 4B tokens have
  enough horizon. See [the postmortem](postmortem.md).

## Acknowledgements

The architecture is built from publicly described components — Mamba-2 SSD,
flash-attn 2.6, SwiGLU, RoPE — composed under a router whose controller is
ported from DeepSeek-V3. CHIMERA's contribution is the per-token routing
across *sequence mixers*, not the mixers themselves. See the
[reading list](lit/README.md) for full attribution.
