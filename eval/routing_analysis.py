"""Routing analysis — answers the spec's hypothesis: which tokens trigger which mode?

The spec promises *interpretable inspection*: you can show which tokens
triggered exact attention. This module gives the primitive.

For an MQAR sequence specifically, we partition positions into:
  - **kv positions** (storage): the (key, value) pairs at the start
  - **query positions** (lookup): where the model must recall the value
  - **other** (e.g., the QUERY separator)

CHIMERA's thesis predicts: *query positions should fire mode-3 (full attention)
more often than kv positions*. The router should learn to upweight exact
attention exactly when exact recall is needed.

For non-MQAR sequences, `analyze_routing` returns the full per-token per-layer
distribution without position interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chimera.data.synthetic import MQARConfig
from chimera.modules.router import MODE_FULL, MODE_IDENTITY, MODE_SSM, MODE_SWA, NUM_MODES


MODE_NAMES = {
    MODE_IDENTITY: "identity",
    MODE_SSM: "ssm",
    MODE_SWA: "swa",
    MODE_FULL: "full",
}


@dataclass
class RoutingTrace:
    """Per-token routing for one batch.

    Attributes:
      hard:   (B, T, L) long — hard mode per token per layer
      soft:   (B, T, L, K) float — soft weights per token per layer
    """

    hard: torch.Tensor
    soft: torch.Tensor


@torch.no_grad()
def analyze_routing(model, input_ids: torch.Tensor, *, router_mode: str = "hard_top1") -> RoutingTrace:
    """Run the model once and collect per-token routing across all layers.

    Returns RoutingTrace with hard/soft both indexed as (B, T, L, ...).
    """
    model.eval()
    out = model.forward_prefill(input_ids, router_mode=router_mode)
    # router_outputs is a list of length L of RouterOutput; each has
    # hard_index (B, T) and weights (B, T, K).
    hard_per_layer = [r.hard_index for r in out.router_outputs]
    soft_per_layer = [r.weights for r in out.router_outputs]
    hard = torch.stack(hard_per_layer, dim=-1)  # (B, T, L)
    soft = torch.stack(soft_per_layer, dim=-2)  # (B, T, L, K)
    return RoutingTrace(hard=hard, soft=soft)


@dataclass
class MQARRoutingReport:
    """Mode usage at query vs. kv vs. other positions, per layer.

    Each entry shape: (L, K) — fraction of tokens in that position class that
    fired each mode in each layer.
    """

    kv_fractions: torch.Tensor       # (L, K)
    query_fractions: torch.Tensor    # (L, K)
    other_fractions: torch.Tensor    # (L, K)
    raw_trace: RoutingTrace


@torch.no_grad()
def analyze_mqar_routing(
    model,
    task: MQARConfig,
    *,
    batch_size: int = 16,
    seed: int = 0,
    router_mode: str = "hard_top1",
    device: str = "cpu",
) -> MQARRoutingReport:
    """Analyze CHIMERA's routing on an MQAR batch.

    Returns kv / query / other-position routing fractions per layer.
    The headline check: `query_fractions[:, MODE_FULL]` should exceed
    `kv_fractions[:, MODE_FULL]` if the router learns the task.
    """
    from chimera.data.synthetic import make_mqar_batch

    ids, _ = make_mqar_batch(task, batch_size=batch_size, seed=seed)
    ids = ids.to(device)
    trace = analyze_routing(model, ids, router_mode=router_mode)

    N, M = task.num_kv_pairs, task.num_queries
    T = task.seq_len
    kv_positions = list(range(0, 2 * N))                       # [0, 2N)
    query_positions = [2 * N + 1 + 2 * i for i in range(M)]    # q_i positions
    answer_positions = [2 * N + 2 + 2 * i for i in range(M)]   # ? positions
    other_positions = sorted(set(range(T)) - set(kv_positions) - set(query_positions) - set(answer_positions))

    def _fractions_at(positions: list[int]) -> torch.Tensor:
        # trace.hard: (B, T, L). Restrict to selected positions, then one-hot mean.
        if not positions:
            return torch.zeros(trace.hard.shape[-1], NUM_MODES)
        sel = trace.hard[:, positions, :]              # (B, |pos|, L)
        oh = torch.nn.functional.one_hot(sel, num_classes=NUM_MODES).float()
        return oh.mean(dim=(0, 1))                     # (L, K)

    return MQARRoutingReport(
        kv_fractions=_fractions_at(kv_positions),
        query_fractions=_fractions_at(query_positions),
        other_fractions=_fractions_at(other_positions),
        raw_trace=trace,
    )


def format_mqar_routing_report(report: MQARRoutingReport) -> str:
    """Pretty-print the report. One section per layer."""
    L, K = report.query_fractions.shape
    lines = []
    for layer in range(L):
        lines.append(f"=== Layer {layer} ===")
        header = "{:<8} ".format("class") + " ".join(f"{MODE_NAMES[k]:>8}" for k in range(K))
        lines.append(header)
        for label, src in [
            ("kv     ", report.kv_fractions[layer]),
            ("query  ", report.query_fractions[layer]),
            ("other  ", report.other_fractions[layer]),
        ]:
            row = f"{label} " + " ".join(f"{f:>8.3f}" for f in src.tolist())
            lines.append(row)
    return "\n".join(lines)
