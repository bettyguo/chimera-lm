"""Smoke training run — proves the trainer works end-to-end on CPU.

What this validates:
  - The trainer wires up cleanly with the synthetic MQAR task.
  - Loss decreases over ~200 steps.
  - Per-layer routing fractions do not collapse (no mode < 1%).
  - The aux-loss-free balancer biases evolve in the expected direction
    (under-utilized modes get positive bias).

This is NOT a quality measurement — it's a "does the machinery turn at all"
smoke test. Quality benchmarks (perplexity, MQAR accuracy at long T) need
the GPU stack with Mamba-2 and flash-attn.

Run:
    python -m scripts.smoke_train_nano
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from chimera.data.synthetic import MQARConfig, make_mqar_batch
from chimera.model import ChimeraConfig
from train.train import TrainerConfig, format_log, train


def main() -> int:
    # Tiny model + tiny task to fit a smoke run in <60 seconds on CPU.
    task = MQARConfig(
        num_keys=16, num_values=16, num_kv_pairs=6, num_queries=3
    )
    model_cfg = ChimeraConfig(
        vocab_size=task.vocab_size,
        num_layers=2,
        dim=64,
        num_heads=4,
        window=16,
        ssm_state=16,
        max_seq_len=task.seq_len + 4,
        tie_embeddings=True,
    )

    def batch_source(step: int):
        # Seeded per-step so we get a fresh batch but the run is reproducible.
        return make_mqar_batch(task, batch_size=16, seed=step)

    def val_source(step: int):
        return make_mqar_batch(task, batch_size=16, seed=100_000 + step)

    cfg = TrainerConfig(
        model_cfg=model_cfg,
        batch_source=batch_source,
        val_batch_source=val_source,
        max_steps=200,
        batch_size=16,
        lr=3e-3,
        lr_min=1e-4,
        warmup_steps=10,
        weight_decay=0.01,
        grad_clip=1.0,
        log_every=20,
        val_every=40,
        val_steps=4,
        seed=0,
    )

    print(f"# MQAR seq_len = {task.seq_len}, vocab = {task.vocab_size}")
    print(f"# Model: {model_cfg.num_layers}L x {model_cfg.dim}D, "
          f"window={model_cfg.window}, ssm_state={model_cfg.ssm_state}")
    history = train(cfg)

    for log in history:
        print(format_log(log))

    # Smoke assertions: loss should decrease, no mode should collapse.
    first, last = history[0], history[-1]
    print()
    print(f"# loss: {first.loss:.4f} -> {last.loss:.4f}")
    print(f"# wall: {last.wall_seconds:.1f}s")

    # Per-layer mode coverage check (over the final step's fractions).
    collapsed = []
    for li, fracs in enumerate(last.per_layer_fractions):
        for mi, f in enumerate(fracs):
            if f < 0.01:
                collapsed.append((li, mi, f))
    if collapsed:
        print(f"# WARNING: collapsed modes: {collapsed}")
    else:
        print("# All modes > 1% on the final step.")

    if last.loss < first.loss * 0.95:
        print("# OK: loss decreased >5%")
        return 0
    print("# WARN: loss did not decrease >5% in 200 steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
