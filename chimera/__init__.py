"""CHIMERA: Conditionally Hybrid Mixture of Exact and Recurrent Attention.

Symbol conventions used throughout:
  B  — batch size
  T  — sequence length
  D  — model hidden dim
  H  — number of attention heads
  Dk — head dim (D / H)
  W  — sliding-window size (mode 2)
  S  — SSM state size (mode 1)
  K  — number of mixing modes (= 4 in v1: identity, SSM, SWA, full)
  L  — number of layers
"""

from chimera.cache import ChimeraCache, ChimeraCacheLayer
from chimera.model import ChimeraConfig, ChimeraLM
from chimera.modules.chimera_block import ChimeraBlock

__all__ = [
    "ChimeraBlock",
    "ChimeraCache",
    "ChimeraCacheLayer",
    "ChimeraConfig",
    "ChimeraLM",
]
