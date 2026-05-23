"""Tests for the MQAR evaluator and routing analysis."""

import torch

from chimera.baselines import BaselineConfig, PureTransformerLM
from chimera.data.synthetic import MQARConfig
from chimera.model import ChimeraConfig, ChimeraLM
from chimera.modules.router import NUM_MODES
from eval.mqar import evaluate_mqar
from eval.routing_analysis import (
    analyze_mqar_routing,
    analyze_routing,
    format_mqar_routing_report,
)


def test_mqar_eval_on_random_chimera():
    """An untrained model on MQAR should score around 1 / num_values."""
    task = MQARConfig(num_keys=16, num_values=16, num_kv_pairs=4, num_queries=2)
    cfg = ChimeraConfig(
        vocab_size=task.vocab_size,
        num_layers=2,
        dim=32,
        num_heads=4,
        window=8,
        ssm_state=8,
        max_seq_len=task.seq_len + 2,
    )
    torch.manual_seed(0)
    model = ChimeraLM(cfg)
    result = evaluate_mqar(model, task, num_batches=2, batch_size=8, seed=0)
    # Random model: accuracy should be near chance (1 / num_values ≈ 0.06) ± a lot.
    # Just verify the eval runs and produces sane numbers.
    assert 0.0 <= result.overall_acc <= 1.0
    assert len(result.per_query_acc) == task.num_queries
    assert result.loss > 0.0  # untrained model has positive loss
    assert result.num_examples == 2 * 8


def test_mqar_eval_on_baseline():
    """The evaluator works on baselines too (no router_outputs)."""
    task = MQARConfig(num_keys=16, num_values=16, num_kv_pairs=4, num_queries=2)
    cfg = BaselineConfig(
        vocab_size=task.vocab_size, num_layers=2, dim=32, num_heads=4, max_seq_len=task.seq_len + 2
    )
    torch.manual_seed(0)
    model = PureTransformerLM(cfg)
    result = evaluate_mqar(model, task, num_batches=2, batch_size=8, seed=0)
    assert 0.0 <= result.overall_acc <= 1.0


def test_analyze_routing_shapes():
    cfg = ChimeraConfig(
        vocab_size=64, num_layers=3, dim=32, num_heads=4, window=8, ssm_state=8, max_seq_len=32
    )
    torch.manual_seed(0)
    model = ChimeraLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    trace = analyze_routing(model, ids)
    assert trace.hard.shape == (2, 16, 3)
    assert trace.soft.shape == (2, 16, 3, NUM_MODES)
    # Soft weights sum to 1 along K.
    assert torch.allclose(trace.soft.sum(dim=-1), torch.ones(2, 16, 3), atol=1e-5)


def test_mqar_routing_report_shapes():
    task = MQARConfig(num_keys=8, num_values=8, num_kv_pairs=4, num_queries=2)
    cfg = ChimeraConfig(
        vocab_size=task.vocab_size,
        num_layers=2,
        dim=32,
        num_heads=4,
        window=8,
        ssm_state=8,
        max_seq_len=task.seq_len + 2,
    )
    torch.manual_seed(0)
    model = ChimeraLM(cfg)
    report = analyze_mqar_routing(model, task, batch_size=4, seed=0)
    L, K = report.query_fractions.shape
    assert L == 2 and K == NUM_MODES
    assert report.kv_fractions.shape == (L, K)
    assert report.other_fractions.shape == (L, K)
    # Fractions per layer should sum to 1 (or 0 if no positions).
    for arr in [report.kv_fractions, report.query_fractions, report.other_fractions]:
        for layer in range(L):
            s = arr[layer].sum().item()
            assert (abs(s - 1.0) < 1e-4) or (abs(s) < 1e-6)


def test_format_mqar_routing_report_runs():
    task = MQARConfig(num_keys=8, num_values=8, num_kv_pairs=4, num_queries=2)
    cfg = ChimeraConfig(
        vocab_size=task.vocab_size,
        num_layers=2,
        dim=32,
        num_heads=4,
        window=8,
        ssm_state=8,
        max_seq_len=task.seq_len + 2,
    )
    torch.manual_seed(0)
    model = ChimeraLM(cfg)
    report = analyze_mqar_routing(model, task, batch_size=4, seed=0)
    s = format_mqar_routing_report(report)
    # Sanity: contains the mode names and layer headers.
    assert "Layer 0" in s and "Layer 1" in s
    assert "query" in s and "kv" in s
