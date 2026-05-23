"""ChimeraLM — end-to-end decoder language model.

Composition: TokenEmbedding -> [ChimeraBlock x L] -> LayerNorm -> Unembedding.

Decode contract (mirrors the per-block contract): for any sequence and any
routing pattern, prefill(x) and step-by-step decode produce identical output
logits to fp64 bit-exactness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from chimera.cache import ChimeraCache
from chimera.modules.chimera_block import BlockConfig, ChimeraBlock
from chimera.modules.router import RouterOutput


@dataclass
class ChimeraConfig:
    """Model-level config.

    The headline sizes in the spec (nano/small/medium/large) instantiate this.
    """

    vocab_size: int = 32_000
    num_layers: int = 4
    dim: int = 256
    num_heads: int = 4
    window: int = 64
    ssm_state: int = 32
    target_dist: tuple[float, ...] = (0.10, 0.60, 0.20, 0.10)
    ffn_mult: float = 8 / 3
    max_seq_len: int = 2048
    rope_base: float = 10_000.0
    tie_embeddings: bool = True

    def block_config(self) -> BlockConfig:
        return BlockConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            window=self.window,
            ssm_state=self.ssm_state,
            target_dist=self.target_dist,
            ffn_mult=self.ffn_mult,
            max_seq_len=self.max_seq_len,
            rope_base=self.rope_base,
        )


@dataclass
class PrefillOutput:
    logits: torch.Tensor  # (B, T, vocab)
    router_outputs: list[RouterOutput] = field(default_factory=list)  # per layer


@dataclass
class DecodeStepOutput:
    logits: torch.Tensor  # (B, vocab)
    router_outputs: list[RouterOutput] = field(default_factory=list)


class ChimeraLM(nn.Module):
    """The end-to-end CHIMERA language model."""

    def __init__(self, cfg: ChimeraConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [ChimeraBlock(cfg.block_config()) for _ in range(cfg.num_layers)]
        )
        self.norm_final = nn.LayerNorm(cfg.dim)
        if cfg.tie_embeddings:
            self.unembed = None
        else:
            self.unembed = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def empty_cache(self, batch: int, *, device=None, dtype=None) -> ChimeraCache:
        """Allocate an empty per-layer cache for autoregressive decode."""
        device = device or self.tok_embed.weight.device
        dtype = dtype or self.tok_embed.weight.dtype
        return ChimeraCache.empty(
            num_layers=self.cfg.num_layers,
            batch=batch,
            kv_shape=(self.cfg.num_heads, self.cfg.dim // self.cfg.num_heads),
            d_ssm=self.cfg.ssm_state,
            window=self.cfg.window,
            device=device,
            dtype=dtype,
        )

    def _project_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.unembed is None:
            return h @ self.tok_embed.weight.t()
        return self.unembed(h)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        *,
        router_mode: str = "soft",
    ) -> PrefillOutput:
        """Full-sequence forward.

        input_ids: (B, T) long
        returns:   PrefillOutput
        """
        h = self.tok_embed(input_ids)  # (B, T, D)
        router_outs: list[RouterOutput] = []
        for layer in self.layers:
            h, r = layer.forward_prefill(h, router_mode=router_mode)
            router_outs.append(r)
        h = self.norm_final(h)
        logits = self._project_logits(h)
        return PrefillOutput(logits=logits, router_outputs=router_outs)

    def forward_decode_step(
        self,
        input_id: torch.Tensor,
        cache: ChimeraCache,
        position: int,
        *,
        router_mode: str = "hard_top1",
    ) -> DecodeStepOutput:
        """Single-token decode.

        input_id: (B,) long — token at the current position
        cache:    ChimeraCache — mutated in place
        position: integer position index for RoPE
        """
        h = self.tok_embed(input_id)  # (B, D)
        router_outs: list[RouterOutput] = []
        for i, layer in enumerate(self.layers):
            h, r = layer.forward_decode_step(
                h, cache[i], position=position, router_mode=router_mode
            )
            router_outs.append(r)
        h = self.norm_final(h)
        logits = self._project_logits(h)
        return DecodeStepOutput(logits=logits, router_outputs=router_outs)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        router_mode: str = "hard_top1",
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Greedy / temperature-sampled generation.

        prompt_ids: (B, T_prompt) long. Returns (B, T_prompt + max_new_tokens) long.

        Implementation: prefill the prompt, then step. For a cleaner path that
        fills the cache from the prefill we'd need to also feed each prompt
        token through the decode-step path; for v1 we just decode-step the
        prompt to build the cache (slower but obviously correct).
        """
        device = prompt_ids.device
        B, T_prompt = prompt_ids.shape
        cache = self.empty_cache(B, device=device, dtype=self.tok_embed.weight.dtype)
        ids = prompt_ids.clone()
        # Step through the prompt to populate the cache.
        for t in range(T_prompt):
            out = self.forward_decode_step(
                ids[:, t], cache, position=t, router_mode=router_mode
            )
        # Now generate.
        for t in range(T_prompt, T_prompt + max_new_tokens):
            logits = out.logits / max(temperature, 1e-6)
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            out = self.forward_decode_step(
                next_id.squeeze(1), cache, position=t, router_mode=router_mode
            )
        return ids


# ----------------------------------------------------------------------
# Convenience factories matching the spec's named configs.
# ----------------------------------------------------------------------
def nano_config() -> ChimeraConfig:
    """~12M params target. Used for CPU / debug runs."""
    return ChimeraConfig(
        vocab_size=32_000,
        num_layers=4,
        dim=256,
        num_heads=4,
        window=64,
        ssm_state=32,
        max_seq_len=1024,
    )


def small_config() -> ChimeraConfig:
    """~125M params target."""
    return ChimeraConfig(
        vocab_size=32_000,
        num_layers=12,
        dim=768,
        num_heads=12,
        window=256,
        ssm_state=64,
        max_seq_len=2048,
    )


def medium_config() -> ChimeraConfig:
    """~350M params target."""
    return ChimeraConfig(
        vocab_size=32_000,
        num_layers=24,
        dim=1024,
        num_heads=16,
        window=512,
        ssm_state=128,
        max_seq_len=4096,
    )


def large_config() -> ChimeraConfig:
    """~1.3B params target."""
    return ChimeraConfig(
        vocab_size=32_000,
        num_layers=24,
        dim=2048,
        num_heads=16,
        window=512,
        ssm_state=128,
        max_seq_len=4096,
    )
