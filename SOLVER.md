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
| MLA (DeepSeek-V2/V3/V4, GLM-4.x) | any | 4096 | 8192 | Latent attention has ~1/N KV cache vs non-MLA, so ubatch can carry more; Lightning Indexer fused op in CachyLLama accelerates the indexer/attention split when subgroup_size_control gates pass |

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

### Context size candidates

The solver reads the model's `context_length` from GGUF metadata (via
`read_gguf_kv.py`) and uses it directly as the starting `SOLVER_CTX_SIZE`.
Phase-1 candidate context sizes are generated dynamically via
`_opt_build_ctx_candidates(context_length)`: the standard sizes
`{1048576, 524288, 262144, 196608, 131072, 98304, 65536}` filtered to
those ≤ the model's max, capped at 1M (llama.cpp hard limit).

q4_0 KV cache is **not** a phase-1 candidate — it is reserved for phase 2
as an absolute last resort. Phase 1 enumerates ctx from largest to smallest
with KV types `{f16, q8_0}` in score order. The first combo that passes
both GPU and system memory checks (including the `min_cache_ram_mib`
leftover check) wins.

If no phase-1 combo fits, phase 2 starts from the most memory-conservative
fallback (`MIN_CTX` + `q4_0/q4_0` with CPU strategy for MoE models) and
applies detune steps in priority order (q8 → q4 → SSD RAM → draft → SSD
→ ubatch → ctx → NGL).

## Combination scoring

For each (strategy, ctx, KV, draft, batch, ubatch) tuple, the solver computes
a score. Higher scores are tried first. The first combo that fits the GPU
budget AND system memory wins.

```
score = strategy_score * 1000      # gpu=300, cpu=250, residency=200
      + ctx_score * 10             # 1048576=110, 524288=105, 262144=100, 196608=95,
                                   # 131072=90, 98304=85, 65536=80
      + kv_score                   # f16=30, q8_0=20
      + draft_score                # enabled=5
      + batchub_score              # opt=2, partial-match=1, other=0
```

The ctx_score is monotonically increasing with context size, so the
solver prefers the largest context that fits. The kv_score prefers f16
over q8_0. q4_0 is NOT in phase 1 — it is a phase-2 detune step only,
applied when no f16 or q8_0 configuration fits at any context size.

Phase 1 enumerates (ctx from model max down to 128k, kv in {f16, q8_0})
in score order. The (batch, ubatch) component is a small tiebreaker (max
+2) so it does not override the ctx/kv preference.

The (batch, ubatch) component is a small tiebreaker (max +2) so it does
not override the ctx/kv preference. The real decision driver is the GPU
memory check + system memory check.

## Detune steps (phase 2)

If no combination fits in phase 1, phase 2 starts from the most
memory-conservative fallback (MIN_CTX + q4_0/q4_0) and applies detune
steps in priority order. The preference is: **q4_0 KV cache only when no
choice, ctx below 128k only when no choice, reducing layers from GPU only
when no choice.**

The phase-2 detune order (first fit wins, each step applies at most once):

1. Reduce KV cache to q8_0 (only downgrades from f16 — skipped if already q8 or q4)
2. Reduce KV cache to q4_0 (only downgrades from f16/q8 — skipped if already q4)
3. Reduce SSD hot/warm RAM by half
4. Drop speculative draft model
5. Drop SSD cache entirely
6. Reduce ubatch by half (clamped at 512)
7. Reduce ctx (cascading values 1048576 -> 524288 -> 262144 -> 196608 -> 131072 -> 98304 -> 65536)
8. Reduce NGL by 10% per step (layers moved from GPU to CPU — absolute last resort)

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
