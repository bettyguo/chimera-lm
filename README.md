# chimera-lm

Reference implementation of **CHIMERA** — Conditionally Hybrid Mixture of
Exact and Recurrent Attention. A decoder-only LM where each token's mixer
(identity / SSM / sliding-window attention / full causal attention) is
chosen by a learned router with auxiliary-loss-free load balancing.

The thesis: the hybridization ratio between attention and SSM should be
*learned per-token and per-layer*, not a fixed design-time hyperparameter.

See [`docs/architecture.md`](docs/architecture.md), [`docs/routing.md`](docs/routing.md),
and [`docs/kv_cache.md`](docs/kv_cache.md) for the contracts each subsystem
implements.

## Status

Phase 1 complete on CPU. All 30 tests pass at fp64 bit-exactness, including
the causal-consistency canary (prefill ≡ step-by-step decode).

Phase 2+ requires a GPU box with CUDA: swap `ToySSM` for Mamba-2 SSD
(`mamba-ssm`), swap pure-PyTorch attention for `flash-attn` 2.6, then train.
See ADRs in [`docs/decisions/`](docs/decisions/) for the swap-in points.

## Documentation site

The full documentation builds as a MkDocs-Material site, deployed to GitHub
Pages via `.github/workflows/docs.yml`. To build locally:

```bash
pip install -e ".[docs]"
mkdocs build --strict     # produces site/
mkdocs serve              # http://127.0.0.1:8000
```

The site bundles the architecture, routing and cache docs, ADRs, the THINK
note, the postmortem, the nano experiment report, and 10 paper-reading stubs.
Accessibility is checked in CI: each built page must have exactly one `<h1>`,
no skipped heading levels, no empty links, alt text on every image, a `<main>`
landmark, a skip-to-content link, and `<html lang>`. See
[`tests/test_site.py`](tests/test_site.py).

## Install (CPU dev)

```bash
pip install -e ".[dev]"
pytest tests/
```

## Install (GPU)

```bash
pip install -e ".[dev,gpu,train]"
```

`gpu` extras (`triton`, `flash-attn`, `mamba-ssm`) are gated to non-Windows
platforms. On a CUDA Linux box they should install; if `nvcc` is missing,
expect compile errors.

## Layout

```
chimera-lm/
├── chimera/
│   ├── modules/         # building blocks (router, ssm, attention, ffn, rope, block)
│   ├── cache.py         # multi-mode KV cache (Resolution A)
│   ├── model.py         # ChimeraLM + named configs (nano/small/medium/large)
│   ├── losses.py        # CE + routing diagnostics
│   ├── utils/profiling.py   # FLOP / KV memory accounting (spec A.4)
│   └── THINK.md         # Phase 1 design rationale
├── tests/               # pytest suite — 30 tests, fp64 parity
└── docs/                # architecture, routing, cache, ADRs
```

## What's intentionally missing (out of Phase 1 scope)

- `train/` — no trainer, no FineWeb-Edu data, no FSDP. Needs GPU + data
  pipeline that doesn't exist on the dev box.
- `eval/` — no MQAR, no needle-in-haystack, no LongBench. These benchmarks
  only make sense once we have trained checkpoints.
- `ablations/` — Phase 5.
- Lit notes (`docs/lit/`) — the spec asks for annotated reading notes on
  10 papers. Those need actual reading, not fabrication. Add them as you
  read.

## The canary

```bash
pytest tests/test_causal_consistency.py -v
```

If this ever fails on `main`, that's a P0. Write a POSTMORTEM; find the
off-by-one; do not silently fix.
