"""Three-way head-to-head on synthetic MQAR.

Trains:
  - CHIMERA-nano with the standard router + aux-free balancer
  - Pure Transformer baseline (same dim/layers)
  - Pure SSM baseline (same dim/layers)

Then evaluates each on a held-out MQAR set and reports accuracy + per-query
breakdown. For CHIMERA, also dumps the per-position routing table so we can
answer: *does the router fire mode-3 more at query positions than at kv
positions?*

This is an investigation script, NOT a production benchmark — toy SSM, no
flash-attn, ~1 minute total wall-clock. Treat directional findings as
signals, not claims.

Run:
    python -m scripts.investigate_mqar_routing
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time

import torch
import torch.nn.functional as F

from chimera.baselines import BaselineConfig, PureSSMLM, PureTransformerLM
from chimera.data.synthetic import MQARConfig, make_mqar_batch
from chimera.model import ChimeraConfig, ChimeraLM
from eval.mqar import evaluate_mqar
from eval.routing_analysis import analyze_mqar_routing, format_mqar_routing_report


def _make_optimizer(model, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() <= 1 or n.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


def _train(
    model,
    batch_source: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    lr: float,
    label: str,
    is_chimera: bool,
):
    optim = _make_optimizer(model, lr)
    model.train()
    losses = []
    t0 = time.time()
    for step in range(steps):
        ids, targets = batch_source(step)
        out = (
            model.forward_prefill(ids, router_mode="soft")
            if is_chimera
            else model.forward_prefill(ids)
        )
        logits = out.logits
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), ignore_index=-100)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == steps - 1:
            print(f"  [{label}] step {step:4d}  loss {loss.item():.4f}  t {time.time()-t0:.1f}s")
    return losses


def main() -> int:
    torch.manual_seed(0)

    # Tiny task and tiny models for a ~minute CPU run.
    task = MQARConfig(num_keys=16, num_values=16, num_kv_pairs=6, num_queries=3)
    train_batch = 16
    eval_batches = 8

    # Shared dimensions across the three models.
    L, D, H = 2, 64, 4

    chimera_cfg = ChimeraConfig(
        vocab_size=task.vocab_size, num_layers=L, dim=D, num_heads=H,
        window=16, ssm_state=16, max_seq_len=task.seq_len + 4,
    )
    baseline_cfg = BaselineConfig(
        vocab_size=task.vocab_size, num_layers=L, dim=D, num_heads=H,
        ssm_state=16, max_seq_len=task.seq_len + 4,
    )

    print(f"# MQAR seq_len={task.seq_len}, vocab={task.vocab_size}, "
          f"L={L} D={D} H={H}")
    print(f"# Training {train_batch}-batch MQAR for 200 steps each (3 models).\n")

    def src(step):
        return make_mqar_batch(task, batch_size=train_batch, seed=step)

    # 1. CHIMERA
    print("=== Training CHIMERA-nano ===")
    chimera = ChimeraLM(chimera_cfg)
    _train(chimera, src, steps=200, lr=3e-3, label="chimera", is_chimera=True)

    # 2. Pure Transformer
    print("\n=== Training Pure Transformer ===")
    transformer = PureTransformerLM(baseline_cfg)
    _train(transformer, src, steps=200, lr=3e-3, label="transformer", is_chimera=False)

    # 3. Pure SSM
    print("\n=== Training Pure SSM ===")
    ssm = PureSSMLM(baseline_cfg)
    _train(ssm, src, steps=200, lr=3e-3, label="ssm", is_chimera=False)

    # Evaluate on held-out batches (different seed offset).
    print("\n=== Held-out MQAR accuracy (seed offset 5000, 8 batches × 16 = 128 examples) ===")
    results = {}
    for name, model, is_chimera in [
        ("CHIMERA   ", chimera, True),
        ("Transformer", transformer, False),
        ("Pure SSM  ", ssm, False),
    ]:
        r = evaluate_mqar(
            model, task,
            num_batches=eval_batches, batch_size=16, seed=5000,
            router_mode="hard_top1" if is_chimera else "soft",
        )
        per_q = ", ".join(f"q{i}={a:.3f}" for i, a in enumerate(r.per_query_acc))
        print(f"  {name}  overall={r.overall_acc:.3f}  loss={r.loss:.3f}  ({per_q})")
        results[name.strip()] = r

    # Routing analysis: where does CHIMERA fire mode-3?
    print("\n=== CHIMERA routing on MQAR (seed 5000) ===")
    report = analyze_mqar_routing(chimera, task, batch_size=16, seed=5000, router_mode="hard_top1")
    print(format_mqar_routing_report(report))

    # Headline test: does query > kv on mode-3?
    print("\n=== Hypothesis: query positions fire mode-3 more than kv positions? ===")
    from chimera.modules.router import MODE_FULL
    L = report.query_fractions.shape[0]
    for layer in range(L):
        q3 = report.query_fractions[layer, MODE_FULL].item()
        k3 = report.kv_fractions[layer, MODE_FULL].item()
        verdict = "YES" if q3 > k3 + 0.01 else "no"
        print(f"  Layer {layer}: mode-3 at queries={q3:.3f}, at kv={k3:.3f}  [{verdict}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
