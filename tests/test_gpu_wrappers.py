"""Tests for GPU-backend wrappers — verifies the import-gating story works.

On a CPU/Windows machine, these wrappers must raise a clear ImportError when
instantiated. On a CUDA box with the gpu extras installed, the import passes
and the wrapper is exercised by the existing tests.

We do NOT test correctness of the wrappers here — that requires CUDA. The
parity tests in `test_causal_consistency.py` are the correctness oracle once
the wrappers are wired into a ChimeraBlock.
"""

import pytest


def test_mamba2_wrapper_raises_on_cpu_without_mamba_ssm():
    pytest.importorskip("torch")
    try:
        import mamba_ssm  # type: ignore[import-not-found]  # noqa: F401
        pytest.skip("mamba_ssm is installed — skipping CPU-only import test")
    except ImportError:
        pass

    from chimera.modules.mamba2_wrapper import Mamba2SSD

    with pytest.raises(ImportError, match="mamba-ssm is not installed"):
        Mamba2SSD(dim=64, state_size=16)


def test_flash_attn_wrapper_raises_on_cpu_without_flash_attn():
    try:
        import flash_attn  # type: ignore[import-not-found]  # noqa: F401
        pytest.skip("flash_attn is installed — skipping CPU-only import test")
    except ImportError:
        pass

    import torch

    from chimera.modules.flash_attn_wrapper import flash_causal_attention

    q = k = v = torch.zeros(1, 4, 2, 8)
    with pytest.raises(ImportError, match="flash-attn is not installed"):
        flash_causal_attention(q, k, v)
