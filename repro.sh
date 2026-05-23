#!/usr/bin/env bash
# Reproduce the CHIMERA Phase-1 + Phase-2-prep deliverables.
#
# Runs in 1–2 minutes on any modern CPU. On GPU boxes, install with the
# `gpu` extra and the same script works (Mamba-2 / flash-attn shims kick in).
set -euo pipefail

echo "=== Installing (CPU-only deps; add [gpu] on CUDA boxes) ==="
pip install -e ".[dev]" 1>/dev/null

echo
echo "=== Test suite (40+ tests, fp64 parity, must be green) ==="
pytest tests/ -v

echo
echo "=== Parameter counts vs spec headlines ==="
python -m scripts.check_param_counts

echo
echo "=== Smoke training: nano on synthetic MQAR (200 steps) ==="
python -m scripts.smoke_train_nano

echo
echo "=== Three-way head-to-head: CHIMERA vs Transformer vs Pure-SSM ==="
python -m scripts.investigate_mqar_routing

echo
echo "=== Done. See experiments/nano_report.md for analysis. ==="
