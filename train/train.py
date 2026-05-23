"""Minimal CHIMERA trainer — single file, no hydra.

Designed to run on CPU for diagnostics; FSDP / bf16 / grad checkpointing live
in the production trainer (Phase 3+). What this file does:

  - AdamW + cosine LR with warmup
  - Per-step logging of: loss, per-layer routing fractions, per-layer entropy,
    aux-free balancer biases
  - Periodic validation (next-token CE on held-out batches)
  - Periodic checkpointing
  - Optional gradient clipping

Use:

    from train.train import TrainerConfig, train
    train(TrainerConfig(...))

For the smoke driver, see `scripts/smoke_train_nano.py`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from chimera.losses import chimera_train_step_loss
from chimera.model import ChimeraConfig, ChimeraLM

Batch = tuple[torch.Tensor, torch.Tensor]  # (input_ids, targets) — targets use -100 mask
BatchSource = Callable[[int], Batch]       # int -> batch given step number


@dataclass
class TrainerConfig:
    model_cfg: ChimeraConfig
    batch_source: BatchSource
    val_batch_source: BatchSource | None = None

    max_steps: int = 200
    batch_size: int = 8
    lr: float = 3e-4
    lr_min: float = 1e-5
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    router_mode_warmup_steps: int = 0  # 0 = always soft; otherwise switch to hard after N steps
    aux_loss_coef: float = 0.0  # default: aux-loss-free (the controller does the work)

    log_every: int = 10
    val_every: int = 0  # 0 = disabled
    val_steps: int = 4
    ckpt_every: int = 0  # 0 = disabled
    ckpt_dir: str | Path = "checkpoints"

    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32  # bf16 only on H100; fp32 on CPU


@dataclass
class TrainStepLog:
    step: int
    loss: float
    lr: float
    per_layer_fractions: list[list[float]]   # [layer][mode] hard fractions
    per_layer_entropy: list[float]           # [layer] soft entropy
    balancer_biases: list[list[float]]       # [layer][mode] aux-free biases
    val_loss: float | None = None
    wall_seconds: float = 0.0


def cosine_with_warmup(step: int, *, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (max_lr - min_lr) * cos


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Standard AdamW with no-decay on biases and 1d params (norms)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _balancer_snapshot(model: ChimeraLM) -> list[list[float]]:
    return [block.router.balancer.bias.tolist() for block in model.layers]


def _router_mode_for(cfg: TrainerConfig, step: int) -> str:
    if cfg.router_mode_warmup_steps <= 0:
        return "soft"
    return "soft" if step < cfg.router_mode_warmup_steps else "hard_top1"


def _validation_loss(
    model: ChimeraLM,
    source: BatchSource,
    num_steps: int,
    router_mode: str,
    device: str,
    dtype: torch.dtype,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(num_steps):
            ids, targets = source(s)
            ids = ids.to(device)
            targets = targets.to(device)
            out = model.forward_prefill(ids, router_mode=router_mode)
            # Use a fresh diagnostic loss call (no aux, balancer not in train).
            tl = chimera_train_step_loss(out.logits, targets, out.router_outputs)
            total += float(tl.ce.detach())
            count += 1
    model.train()
    return total / max(count, 1)


def train(cfg: TrainerConfig) -> list[TrainStepLog]:
    """Run training; return per-log-step records."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ChimeraLM(cfg.model_cfg).to(device=device, dtype=cfg.dtype)
    optim = _build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[TrainStepLog] = []
    ckpt_dir = Path(cfg.ckpt_dir)
    if cfg.ckpt_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model.train()
    for step in range(cfg.max_steps):
        # LR schedule
        lr = cosine_with_warmup(
            step,
            warmup=cfg.warmup_steps,
            total=cfg.max_steps,
            max_lr=cfg.lr,
            min_lr=cfg.lr_min,
        )
        for pg in optim.param_groups:
            pg["lr"] = lr

        # Data
        ids, targets = cfg.batch_source(step)
        ids = ids.to(device)
        targets = targets.to(device)

        # Forward
        router_mode = _router_mode_for(cfg, step)
        out = model.forward_prefill(ids, router_mode=router_mode)
        tl = chimera_train_step_loss(
            out.logits, targets, out.router_outputs, aux_loss_coef=cfg.aux_loss_coef
        )

        # Backward + step
        optim.zero_grad(set_to_none=True)
        tl.loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        # Log every N steps
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            val_loss = None
            if cfg.val_every > 0 and cfg.val_batch_source is not None and step % cfg.val_every == 0:
                val_loss = _validation_loss(
                    model, cfg.val_batch_source, cfg.val_steps, router_mode, cfg.device, cfg.dtype
                )
            log = TrainStepLog(
                step=step,
                loss=float(tl.loss.detach()),
                lr=lr,
                per_layer_fractions=[f.tolist() for f in tl.per_layer_fractions],
                per_layer_entropy=[float(e) for e in tl.per_layer_entropy],
                balancer_biases=_balancer_snapshot(model),
                val_loss=val_loss,
                wall_seconds=time.time() - start,
            )
            history.append(log)

        # Checkpoint
        if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
            path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save(
                {"model": model.state_dict(), "step": step, "cfg": cfg.model_cfg},
                path,
            )

    return history


def format_log(log: TrainStepLog) -> str:
    """Format a single TrainStepLog as a human-readable line."""
    fracs = "; ".join(
        "L{}=[{}]".format(li, ",".join(f"{f:.2f}" for f in lf))
        for li, lf in enumerate(log.per_layer_fractions)
    )
    s = (
        f"step {log.step:5d} | loss {log.loss:.4f} | lr {log.lr:.2e} | "
        f"t {log.wall_seconds:.1f}s | fracs {fracs}"
    )
    if log.val_loss is not None:
        s += f" | val {log.val_loss:.4f}"
    return s
