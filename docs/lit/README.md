# Literature notes

Per spec §1, every CHIMERA implementer must read and annotate these papers
**before writing the corresponding subsystem**. Each note must record:

- The **primitive operation** the paper defines.
- The **computational complexity** of that primitive.
- The **failure mode** the paper *itself* admits.
- The **empirically strongest result** reported.

The 10 stubs in this directory are *templates only* — the implementer fills
them in while reading. Do not fabricate content. An unread paper produces
worse code than no paper at all.

## Required reading

| # | Paper | Used by | Subsystem |
|---|---|---|---|
| 1 | Gu & Dao 2023 — Mamba (selective SSM)               | mode 1            | `chimera/modules/ssm.py` |
| 2 | Dao & Gu 2024 — Mamba-2 / SSD                        | mode 1 (target)   | `chimera/modules/ssm.py` |
| 3 | Lieber et al. 2024 — Jamba                          | hybrid baseline   | `ablations/fixed_ratio_baseline.py` |
| 4 | Glorioso et al. 2024 — Zamba; Ren et al. 2024 — Samba | hybrid baselines | `ablations/` |
| 5 | Raposo et al. 2024 — Mixture-of-Depths              | router prior art  | `chimera/modules/router.py` |
| 6 | Arora et al. 2024 — Zoology / Based                 | MQAR benchmark    | `eval/mqar.py` |
| 7 | Waleffe et al. 2024 — Empirical Mamba scaling laws   | scaling plan      | `scripts/scaling_laws.py` |
| 8 | Fedus et al. 2022 — Switch Transformer              | aux-loss baseline | `chimera/losses.py::load_balance_aux_loss` |
| 9 | DeepSeek-AI 2024 — DeepSeek-V3                       | aux-free balancer | `chimera/modules/router.py::AuxFreeBalancer` |
| 10 | Peng et al. 2024 — RWKV-7                          | linear RNN ref    | comparison only |
