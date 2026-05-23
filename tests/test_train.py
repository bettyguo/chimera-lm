"""Smoke tests for the trainer — runs a couple of steps without exploding."""


from chimera.data.synthetic import MQARConfig, make_mqar_batch
from chimera.model import ChimeraConfig
from train.train import TrainerConfig, cosine_with_warmup, train


def test_cosine_with_warmup_monotonic_warmup():
    lrs = [
        cosine_with_warmup(s, warmup=5, total=100, max_lr=1.0, min_lr=0.0)
        for s in range(5)
    ]
    # Strictly increasing during warmup.
    for i in range(1, len(lrs)):
        assert lrs[i] > lrs[i - 1]


def test_cosine_with_warmup_post_warmup_decays():
    """After warmup, LR should be non-increasing until it hits min."""
    lrs = [
        cosine_with_warmup(s, warmup=5, total=100, max_lr=1.0, min_lr=0.1)
        for s in range(10, 100)
    ]
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-6


def test_cosine_hits_min_at_end():
    lr = cosine_with_warmup(100, warmup=5, total=100, max_lr=1.0, min_lr=0.1)
    assert abs(lr - 0.1) < 1e-6


def test_trainer_runs_a_few_steps_and_decreases_loss():
    """5 training steps on tiny MQAR; loss at step 4 should be < step 0."""
    task = MQARConfig(num_keys=8, num_values=8, num_kv_pairs=4, num_queries=2)
    model_cfg = ChimeraConfig(
        vocab_size=task.vocab_size,
        num_layers=2,
        dim=32,
        num_heads=2,
        window=8,
        ssm_state=8,
        max_seq_len=task.seq_len + 2,
    )

    def src(step):
        return make_mqar_batch(task, batch_size=8, seed=step)

    cfg = TrainerConfig(
        model_cfg=model_cfg,
        batch_source=src,
        max_steps=20,
        lr=1e-2,
        warmup_steps=2,
        log_every=1,
        seed=0,
    )
    history = train(cfg)
    # 20 logs (every step), loss should decrease over the window.
    assert len(history) == 20
    early = sum(h.loss for h in history[:5]) / 5
    late = sum(h.loss for h in history[-5:]) / 5
    assert late < early, f"loss did not decrease: {early:.4f} -> {late:.4f}"


def test_balancer_responds_to_observation():
    """Run a few train steps and verify balancer biases moved off zero."""
    task = MQARConfig(num_keys=8, num_values=8, num_kv_pairs=4, num_queries=2)
    model_cfg = ChimeraConfig(
        vocab_size=task.vocab_size,
        num_layers=2,
        dim=32,
        num_heads=2,
        window=8,
        ssm_state=8,
        max_seq_len=task.seq_len + 2,
    )

    def src(step):
        return make_mqar_batch(task, batch_size=8, seed=step)

    cfg = TrainerConfig(
        model_cfg=model_cfg,
        batch_source=src,
        max_steps=30,
        lr=1e-2,
        warmup_steps=2,
        log_every=29,  # only log start + end
        seed=0,
    )
    history = train(cfg)
    # At least one bias entry in some layer should have moved away from zero.
    final = history[-1].balancer_biases
    moved = any(abs(b) > 1e-4 for layer in final for b in layer)
    assert moved, "balancer biases never updated"
