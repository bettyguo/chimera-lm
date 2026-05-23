"""MQAR accuracy evaluator.

Given a model with `forward_prefill(input_ids)` and an MQARConfig, generate
batches and report:

  - **per-query accuracy**: fraction of query positions where argmax matches the
    target value
  - **per-position accuracy**: accuracy bucketed by query index 0..M-1, since
    later queries see longer context
  - **loss** at query positions only

Works with `ChimeraLM`, `PureTransformerLM`, `PureSSMLM` — any module whose
`forward_prefill` returns an object with a `.logits` attribute of shape
(B, T, V).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chimera.data.synthetic import MQARConfig, make_mqar_batch


@dataclass
class MQARResult:
    overall_acc: float
    per_query_acc: list[float]   # length M
    loss: float
    num_queries: int
    num_examples: int


@torch.no_grad()
def evaluate_mqar(
    model,
    task: MQARConfig,
    *,
    num_batches: int = 8,
    batch_size: int = 16,
    seed: int = 0,
    device: str = "cpu",
    router_mode: str = "hard_top1",
) -> MQARResult:
    """Run MQAR accuracy / loss over `num_batches` random batches.

    `router_mode` is only consumed by ChimeraLM; baselines ignore it.
    """
    model.eval()
    M = task.num_queries
    N = task.num_kv_pairs
    base = 2 * N + 1
    query_positions = [base + 2 * i for i in range(M)]

    correct = torch.zeros(M, dtype=torch.long)
    total_per_query = 0
    loss_total = 0.0
    loss_count = 0

    for b in range(num_batches):
        ids, targets = make_mqar_batch(task, batch_size=batch_size, seed=seed + b)
        ids = ids.to(device)
        targets = targets.to(device)
        out = model.forward_prefill(ids, router_mode=router_mode)
        logits = out.logits  # (B, T, V)

        # Per-query accuracy.
        for i, pos in enumerate(query_positions):
            preds = logits[:, pos].argmax(dim=-1)  # (B,)
            true = targets[:, pos]                  # (B,)
            correct[i] += (preds == true).sum().item()
        total_per_query += batch_size

        # Cross-entropy at valid (non -100) positions.
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        mask = flat_targets != -100
        if mask.any():
            ce = F.cross_entropy(flat_logits[mask], flat_targets[mask], reduction="sum")
            loss_total += float(ce.item())
            loss_count += int(mask.sum().item())

    per_q = (correct.float() / total_per_query).tolist()
    overall = sum(per_q) / max(M, 1)
    return MQARResult(
        overall_acc=overall,
        per_query_acc=per_q,
        loss=loss_total / max(loss_count, 1),
        num_queries=M,
        num_examples=num_batches * batch_size,
    )
