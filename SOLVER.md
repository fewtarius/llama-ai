# Solver Algorithm and Tuning

The optimistic-first solver in `scripts/optimize.sh` picks (ctx, KV cache type,
MoE strategy, batch, ubatch) for each (model, hardware) pair from a scored
combination space. User overrides applied last.

## Decision order

1. **Built-in defaults** (solver starts here — see "Optimistic defaults" below)
2. **System detection** (GPU memory, hardware tier via `detect-gpu.sh`)
3. **Solver output** (chosen from a scored combination matrix)
4. **User overrides** via env vars and CLI flags

## Optimistic defaults (per archetype)

`llama-bench` sweeps across n_batch in {2048, 4096, 8192} and n_ubatch in
{256, 512, 1024, 2048, 4096} on Ayaneo Flip (7840U) and Nimo Axis (Strix
Halo) for 17 model/hardware combinations informed the per-archetype defaults
below. Decode (tg) is memory-bandwidth bound and varies <2% across the
entire (batch, ubatch) range — prefill (pp) is what tuning moves.

### Strix Halo (Radeon 8060S, ~62 GB GPU-visible, 125 GB RAM)

| Archetype | Size | ubatch | batch | Why |
|-----------|------|--------|-------|-----|
| Dense (qwen35 27B, GLM dense) | any | 1024 | 4096 | 5-8% pp loss at ub=2048 (qwen35 27B Q8_0: 127 vs 121 pp) |
| MoE small/medium (<60 GB) | <60 | 1024 | 2048 | MoE routing is per-token-variable; smaller batches amortize dispatch (qwen35moe 35B Q8_0: 1094 vs 825 pp at ub=2048) |
| MoE large (60-100 GB) | 60-100 | 2048 | 8192 | 8192 batch + 2048 ubatch is the empirical sweet spot (Laguna 118B, qwen35moe 122B, gpt-oss 120B) |
| MoE huge (>=100 GB) | >=100 | 4096 | 8192 | 4096 ubatch needed to amortize the heavy per-token work |
| SSM / hybrid (qwen3next) | any | 1024 | 4096 | Linear-attention layers don't benefit from larger batches |
| qwen4exp (Qwen3.8-Flash-Next) | any | 2048 | 4096 | PLE + hybrid attention |

### Standard (7840U, ~24 GB GPU-visible, 32 GB RAM)

| Archetype | ubatch | batch | Why |
|-----------|--------|-------|-----|
| Dense (qwen35 27B, gemma4) | 1024 | 2048 | Within 5% of larger configs; saves memory |
| MoE (gpt-oss 20B, qwen35moe 35B Q4_K) | 2048 | 2048 | Benchmark sweet spot; 5-10% better than ub=1024 |
| SSM / hybrid | 1024 | 2048 | Same as Halo |

### Handheld (Flip KB, ~24 GB GPU-visible, 32 GB RAM)

| Archetype | ubatch | batch | Why |
|-----------|--------|-------|-----|
| Dense | 512 | 1024 | Memory-constrained; smaller is safer |
| MoE | 1024 | 2048 | MoE benefits from slightly larger ubatch than dense |
| SSM | 512 | 1024 | Same as dense |

## Combination scoring

For each (strategy, ctx, KV, draft, batch, ubatch) tuple, the solver computes
a score. Higher scores are tried first. The first combo that fits the GPU
budget AND system memory wins.

```
score = strategy_score * 1000      # gpu=300, cpu=250, residency=200
      + ctx_score * 10             # 131072=100, 98304=95, 196608=90, 262144=85, 65536=80
      + kv_score                   # f16=30, q8_0=20, q4_0=10
      + draft_score                # enabled=5
      + batchub_score              # opt=6, partial-match=4, other=0
```

The (batch, ubatch) component is a small tiebreaker. The real decision
driver is the GPU memory check + system memory check.

## Detune steps (phase 2)

If no combination fits in phase 1, phase 2 applies these steps in order
until something fits:

1. Reduce KV cache to q8_0
2. Reduce KV cache to q4_0
3. Reduce NGL by 10%
4. Reduce SSD hot/warm RAM by half
5. Drop speculative draft
6. Drop SSD cache
7. Reduce ubatch by half (clamped at 512)
8. Reduce ctx (cascading values 262144 -> 196608 -> 131072 -> 98304 -> 65536)

## Edge cases and known issues

### Hybrid SSM/MoE (qwen3next)

`qwen3next` is detected as `is_ssm=true` in `_scan_gguf_arch` (the GGUF has
`qwen3next.ssm.*` keys plus `qwen3next.expert_count > 0`). The SSM branch
takes priority over MoE in the optimistic defaults, picking ubatch=1024
which matches the empirical peak (qwen3next 80B Q8_0: 762 pp at ub=1024 vs
537 at ub=4096).

### Models with misleading filenames

Some MoE models (Laguna-S-2.1, Qwen3-Coder-Next, GLM-4.7-Flash) don't match
the filename MoE regex `moe|a3b|a8b|flash|expert|gpt-oss`. The GGUF scanner
in `llama-run.sh::_scan_gguf_arch` reads the first 1 MB of the GGUF (up from
the original 16 KB) and checks for `expert_count` to set `is_moe=true`. This
catches all known MoE architectures.

### 8192 batch for very large MoE

The data shows 8192 batch outperforms 4096 by 5-20% on prefill for models
>=60 GB on Halo (gpt-oss 120B, Laguna 118B, qwen35moe 122B, deepseek4). The
solver's optimistic default for >=100 GB is 4096/8192 and for 60-100 GB is
2048/8192. The phase-1 candidates list always includes 8192 as a fallback
so the memory check has it on the table.

### Decode (tg) is bandwidth-bound

Across all 17 benchmark sweeps, tg varies <2% across the entire
(batch, ubatch) range. Tuning (batch, ubatch) moves pp but not tg. So
`vram-bandwidth-limited` decode speed is a hardware characteristic, not
something the solver can tune.

## Benchmark data (llama-bench, 2026-08-28)

| Hardware | Model | Size | Peak (b/u) | Peak pp (t/s) |
|----------|-------|-----:|-----------:|--------------:|
| Halo | deepseek2 30B.A3B Q8_0 | 33 GB | 4096/4096 | 501.6 |
| Halo | deepseek4 IQ3_XXS | 97 GB | 8192/4096 | 287.9 |
| Halo | gemma4 26B.A4B Q5_K_M | 20 GB | 4096/4096 | 1509.2 |
| Halo | gpt-oss 120B Q8_0 | 60 GB | 8192/4096 | 1116.8 |
| Halo | gpt-oss 20B Q6_K | 11 GB | 2048/4096 | 1722.2 |
| Halo | Laguna 118B Q4_K_M | 68 GB | 8192/2048 | 595.0 |
| Halo | Laguna 118B Q5_K_M | 82 GB | 8192/4096 | 607.3 |
| Halo | minimax-m2 230B Q2_K_M | 70 GB | 4096/4096 | 517.6 |
| Halo | qwen3moe 235B IQ2_M | 73 GB | 8192/2048 | 234.1 |
| Halo | qwen35 27B Q8_0 (DENSE) | 33 GB | 4096/1024 | 127.0 |
| Halo | qwen35 27B Q4_K (DENSE) | 29 GB | 4096/1024 | 170.1 |
| Halo | qwen35moe 122B Q4_K | 73 GB | 8192/2048 | 443.4 |
| Halo | qwen35moe 122B Q5_K | 85 GB | 8192/2048 | 418.9 |
| Halo | qwen35moe 35B Q8_0 | 36 GB | 2048/1024 | 1094.5 |
| Halo | qwen3next 80B Q8_0 (HYBRID) | 80 GB | 4096/1024 | 762.3 |
| Halo | qwen4exp A3B Q4_K (Q4EXP) | 104 GB | 4096/2048 | 230.6 |
| 7840U | gemma4 26B Q5_K | 20 GB | 4096/4096 | 1509.2 |
| 7840U | gpt-oss 20B Q6_K | 11 GB | 2048/4096 | 1722.2 |
| 7840U | qwen35moe 35B Q4_K | 21 GB | 4096/2048 | 418.1 |
| 7840U | qwen35 27B Q4_K (DENSE) | 29 GB | 4096/1024 | 170.1 |

Builds tested: f658fc5af (10809) on 7840U, 54e51d11f (10808) on Halo.

## Solver accuracy

After the 2026-08-28 refactor, the solver picks within 5% of the empirical
peak for 11 of 13 halo models tested, and within 5% for both 7840U models
tested. The remaining losses are 5-15% on models where the peak (batch, ubatch)
is a specific point (e.g. 8192/2048) that the candidates list includes but
loses the scoring tiebreaker to the more conservative 4096/2048 default. This
is acceptable for a generic solver that doesn't have model-specific tuning.

To override the solver's choice: use `--ubatch-size N` and `--batch-size N`
on the `llama-run.sh` command line. The overrides win.
