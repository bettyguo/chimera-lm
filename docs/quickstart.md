# Quickstart

## Install (CPU dev box)

```bash
git clone https://github.com/bettyguo/chimera-lm.git
cd chimera-lm
pip install -e ".[dev]"
```

Everything from this point runs without CUDA.

## Install (GPU box)

```bash
pip install -e ".[dev,gpu,train]"
```

The `gpu` extra pulls in `triton`, `flash-attn`, and `mamba-ssm`. They will
fail to build on Windows or without `nvcc`; install on a CUDA Linux box.

## Run the test suite

```bash
pytest tests/ -v
```

Expect **51 passing tests** in 4–5 seconds. The headline group:

```text
tests/test_causal_consistency.py — prefill ≡ decode parity (fp64)
tests/test_router.py             — router shapes + aux-free balancer
tests/test_eval.py               — MQAR accuracy + routing analysis
tests/test_train.py              — trainer + cosine LR + balancer updates
```

If `test_causal_consistency` ever fails on `main`, that's a P0. Write a
[postmortem](postmortem.md) before fixing.

## Reproduce the smoke run

```bash
bash repro.sh
```

This will:

1. Install dependencies.
2. Run the test suite (~5 s).
3. Print parameter counts for the four named configs.
4. Train CHIMERA-nano on synthetic MQAR for 200 steps (~10 s).
5. Run the three-way head-to-head (CHIMERA, Pure Transformer, Pure SSM)
   with routing analysis (~10 s).

Total wall-clock: ~30 seconds on CPU. Output mirrors the [nano report](experiments/nano_report.md).

## Use the model directly

```python
import torch
from chimera.model import ChimeraConfig, ChimeraLM

cfg = ChimeraConfig(vocab_size=32_000, num_layers=4, dim=256, num_heads=4)
model = ChimeraLM(cfg)
model.eval()

ids = torch.randint(0, cfg.vocab_size, (1, 64))

# Prefill: whole-sequence forward.
out = model.forward_prefill(ids, router_mode="hard_top1")
print(out.logits.shape)  # torch.Size([1, 64, 32000])

# Step-by-step decode from an empty cache.
cache = model.empty_cache(batch=1)
for t in range(64):
    step = model.forward_decode_step(ids[:, t], cache, position=t)
# Prefill and decode produce bit-identical logits (fp64); see
# `tests/test_causal_consistency.py`.
```

## Inspect routing on a batch

```python
from chimera.data.synthetic import MQARConfig
from eval.routing_analysis import analyze_mqar_routing, format_mqar_routing_report

task = MQARConfig(num_keys=16, num_values=16, num_kv_pairs=6, num_queries=3)
report = analyze_mqar_routing(model, task, batch_size=8, seed=0)
print(format_mqar_routing_report(report))
```

## Folder layout

```text
chimera-lm/
├── chimera/                  # the model
│   ├── modules/              # router, ssm, attention, ffn, rope, block
│   ├── cache.py              # multi-mode KV cache
│   ├── model.py              # ChimeraLM + named configs
│   ├── losses.py             # CE + routing diagnostics
│   ├── baselines.py          # PureTransformerLM, PureSSMLM
│   ├── data/synthetic.py     # MQAR + selective-copy
│   └── utils/profiling.py    # spec A.4 FLOP / memory formulas
├── train/train.py            # AdamW + cosine + per-step logging
├── eval/                     # mqar.py + routing_analysis.py
├── scripts/                  # smoke run, investigation, param check
├── tests/                    # 51 tests, all green
├── experiments/              # nano_report.md
└── docs/                     # this site
```
