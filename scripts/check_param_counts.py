"""Print parameter counts for the named CHIMERA configs.

The spec calls the four sizes nano (~12M), small (~125M), medium (~350M),
large (~1.3B). This script verifies that our current configs roughly match,
so reviewers can see at a glance where the param budget goes.

Run:
    python -m scripts.check_param_counts
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from chimera.model import ChimeraLM, large_config, medium_config, nano_config, small_config


def fmt(n: int) -> str:
    for unit, div in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def param_breakdown(model: ChimeraLM) -> dict[str, int]:
    out: dict[str, int] = {"embedding": 0, "blocks": 0, "final_norm": 0, "unembed": 0}
    out["embedding"] = sum(p.numel() for p in model.tok_embed.parameters())
    for b in model.layers:
        out["blocks"] += sum(p.numel() for p in b.parameters())
    out["final_norm"] = sum(p.numel() for p in model.norm_final.parameters())
    if model.unembed is not None:
        out["unembed"] = sum(p.numel() for p in model.unembed.parameters())
    out["total"] = sum(p.numel() for p in model.parameters())
    return out


def main() -> None:
    rows = [
        ("nano", nano_config(), 12),
        ("small", small_config(), 125),
        ("medium", medium_config(), 350),
        ("large", large_config(), 1300),
    ]
    print(f"{'config':<8}  {'total':>8}  {'embedding':>10}  {'blocks':>10}  {'spec(M)':>8}")
    print("-" * 56)
    for name, cfg, target_M in rows:
        model = ChimeraLM(cfg)
        b = param_breakdown(model)
        print(
            f"{name:<8}  {fmt(b['total']):>8}  {fmt(b['embedding']):>10}  "
            f"{fmt(b['blocks']):>10}  {target_M:>8}"
        )
        del model


if __name__ == "__main__":
    main()
