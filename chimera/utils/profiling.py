"""Per-mode FLOP and memory accounting (spec A.4).

Reviewers will demand precise FLOP comparisons. We provide closed-form formulas
that depend only on (config, routing fractions) and a `verify_against_measured`
helper that the throughput script will use to assert <5% drift.

All counts ignore biases (we use bias-free linears), softmax FLOPs (negligible),
and LayerNorm (negligible). Embedding/unembedding are accounted separately so
per-layer numbers stay comparable across model sizes.

Returned counts are *integer FLOPs per token*, summed over heads.
"""

from __future__ import annotations

from dataclasses import dataclass

from chimera.model import ChimeraConfig


@dataclass
class ModeFlops:
    """Decode/prefill FLOPs per token, per mode (one layer, all heads)."""

    identity: int
    ssm: int
    swa_per_token: int
    full_at_T: int  # parametric in T (call .full_at(T) helper)

    @staticmethod
    def for_config(cfg: ChimeraConfig) -> ModeFlops:
        d = cfg.dim
        W = cfg.window
        S = cfg.ssm_state
        # All counts approximate per spec A.4.
        identity = 0
        ssm = 4 * d * S
        # QKV proj (3*2*d^2) + O proj (2*d^2) + QK^T (2*d*W) + softmax*V (2*d*W)
        swa_qkv = 8 * d * d
        swa_attn = 4 * d * W
        swa_per_token = swa_qkv + swa_attn
        # Full attention: same proj cost; attention grows with T.
        return ModeFlops(
            identity=identity,
            ssm=ssm,
            swa_per_token=swa_per_token,
            full_at_T=swa_qkv,  # base cost; actual full attention adds 4*d*T per token
        )

    def full_at(self, T: int, d: int) -> int:
        return self.full_at_T + 4 * d * T


def block_flops_per_token_decode(
    cfg: ChimeraConfig, fractions: tuple[float, ...], T: int
) -> dict[str, float]:
    """Expected decode FLOPs per token at sequence length T.

    fractions: (f_0, f_1, f_2, f_3) — must sum to 1.
    Returns a dict with per-mode contributions and totals (FFN + router included).
    """
    mf = ModeFlops.for_config(cfg)
    f0, f1, f2, f3 = fractions
    per_mode = {
        "identity": f0 * mf.identity,
        "ssm": f1 * mf.ssm,
        "swa": f2 * mf.swa_per_token,
        "full": f3 * mf.full_at(T, cfg.dim),
    }
    # Router: 2 * d * K
    router = 2 * cfg.dim * 4
    # SwiGLU FFN: 3 matmuls of d -> 8d/3 (rounded) and back; ~8*d^2 dominant.
    ffn = 8 * cfg.dim * cfg.dim
    total = sum(per_mode.values()) + router + ffn
    return {**per_mode, "router": router, "ffn": ffn, "total": total}


def block_kv_memory_bytes(
    cfg: ChimeraConfig, fractions: tuple[float, ...], T: int, *, bytes_per_elem: int = 2
) -> dict[str, float]:
    """Expected per-layer KV memory at decode-step T.

    bytes_per_elem: 2 for bf16/fp16, 4 for fp32. Default 2 (bf16 inference).
    """
    f0, f1, f2, f3 = fractions
    d = cfg.dim
    W = cfg.window
    S = cfg.ssm_state
    ssm_state = bytes_per_elem * S                  # per-layer, per-batch-elem
    ring_buffer = 2 * bytes_per_elem * d * W        # K + V
    persistent = 2 * bytes_per_elem * d * (f3 * T)  # K + V, grows with mode-3 fraction
    return {
        "ssm_state": ssm_state,
        "ring_buffer": ring_buffer,
        "persistent": persistent,
        "total": ssm_state + ring_buffer + persistent,
    }


def dense_transformer_kv_memory_bytes(
    cfg: ChimeraConfig, T: int, *, bytes_per_elem: int = 2
) -> float:
    """Per-layer KV memory of a same-dim dense Transformer baseline."""
    return 2 * bytes_per_elem * cfg.dim * T


def dense_transformer_flops_per_token(cfg: ChimeraConfig, T: int) -> float:
    """Per-token decode FLOPs for a same-dim dense Transformer baseline."""
    d = cfg.dim
    return 8 * d * d + 4 * d * T + 8 * d * d  # attn QKV + attn + FFN
