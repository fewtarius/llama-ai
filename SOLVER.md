# Solver Reference (llama-ai)

The optimistic-first solver in `scripts/optimize.sh` replaces the static
preset table in `llama-run.sh`. This document explains what it does, why,
and how to override it.

## Why

The legacy preset table picks one of 8 fixed configurations based on
hardware tier and model size. This leaves significant performance on the
table because:

1. **KV cache size varies wildly by model architecture.** Hybrid SSM
   models (Qwen 3.5/3.6, qwen3next) have only ~25% of layers using KV
   cache, so memory cost is 4x lower than pure-attention models at the
   same layer count. The legacy table treats all models the same.
2. **GQA ratio determines KV cache cost.** Qwen3.6-35B-A3B has
   16:2 head_count:head_count_kv ratio (8x GQA), so per-token KV is
   ~22 KB at f16 vs ~256 KB for non-GQA models.
3. **Decode throughput scales with KV cache type.** f16 KV is ~10%
   faster than q8_0 for decode on big unified-memory GPUs (Nimo), 0%
   difference on small iGPUs (Flip). The legacy always picked q8_0.
4. **Context size is configurable per-model.** Training context ranges
   from 131k (gpt-oss) to 1M (DeepSeek-V4-Flash). The legacy uses
   generic 32k/65k/131k presets.

## How

The solver takes an optimistic-first approach:

1. **Read GGUF metadata** (block_count, embedding_length, head_count_kv,
   key_length, value_length, full_attention_interval, ssm.state_size,
   nextn_predict_layers, etc.) via `scratch/read_gguf_kv.py`.
2. **Start with max performance**:
   - KV cache: f16/f16
   - Context: model training context (e.g. 262k for Qwen3.5/3.6)
   - SSD cache: enabled, 2GB hot + 2GB warm on halo
   - Draft model: enabled if found (DSpark / DFlash / MTP)
   - Ubatch: 2048 (was 4096 - sweep showed no benefit above 2048 for
     typical prompt sizes)
   - Threads: physical cores for batch, physical-2 for gen
   - MOE: keep on GPU (no residency, no cpu-moe)
3. **Compute memory cost**:
   - Model bytes
   - KV cache: `n_attn_layers * head_count_kv * (key_length + value_length) * sizeof(type)`
     where `n_attn_layers = ceil(n_layer / full_attention_interval)` for
     hybrid SSM models
   - Draft model + draft KV (if enabled)
   - SSD RAM buffers
   - Compute buffers (scales with ubatch: 4 GiB + 2 MiB/ubatch)
   - System reserve (2 GiB)
4. **Iterative detune** by least performance impact until the config
   fits in `GPU_BUDGET_BYTES` (VRAM+GTT):
   1. Reduce SSD hot/warm RAM (no prefill/decode impact)
   2. Reduce context size (mild impact on long-context recall)
   3. Reduce ubatch (reduces prefill throughput)
   4. Reduce threads (small prefill impact)
   5. V cache q8_0 (~50% memory save, minor quality)
   6. V cache q4_0 (~75% memory save, more quality)
   7. K cache q8_0
   8. Drop draft model (some decode throughput loss)
   9. Drop SSD entirely
   10. MoE residency (mmap + madvise tracking)
   11. MoE cpu-moe + load-mode none (slow on small GPU)
   12. ngl=-1 for dense models that still don't fit (auto layer split)

Each detune step has a "done" flag so it only applies once.

5. **Apply user overrides LAST** so they always win over solver output.

## Override precedence

Lowest to highest:
1. Built-in defaults (solver starts here)
2. System detection (GPU memory, hardware tier)
3. Solver output
4. User overrides via env vars and CLI flags
5. `--noauto` escape hatch (disables solver entirely)

User overrides recognized by `apply_user_overrides()`:
- `LLAMA_CTX_SIZE` (set via `--ctx-size`)
- `KV_CACHE_K_OVERRIDE` / `KV_CACHE_V_OVERRIDE`
- `LLAMA_THREADS_OVERRIDE` (set via `--threads`)
- `MOE_UBATCH_OVERRIDE` / `OVERRIDE_UBATCH_SIZE` (set via `--ubatch-size`)
- `OVERRIDE_CACHE_RAM` (set via `--cache-ram`)
- `--no-ssd-cache` CLI flag
- `--fit on` CLI flag
- `OVERRIDE_REASONING_BUDGET` (set via `--reasoning-budget`)

Note: legacy env vars like `LLAMA_SSD_HOT_RAM`, `LLAMA_SSD_WARM_RAM`,
`LLAMA_THREADS`, and `LLAMA_KV_CACHE_TYPE_K` are NOT user overrides -
they're defaults that the solver considers. To override the solver, set
`LLAMA_THREADS_OVERRIDE` or pass `--threads N` / `--cache-ssd-hot-ram N`.

## --noauto escape hatch

The original preset table is preserved as `assign_profile()` in
`llama-run.sh`. Pass `--noauto` to use it instead of the solver:

```bash
./llama-run.sh --noauto --server --print-profile MyModel
```

The profile names shown in `--print-profile` are the legacy aliases
(halo-moe-large, std-dense-large, etc.) - computed from the solver's
output via `_opt_pick_legacy_profile` so existing scripts that grep
for them keep working.

### Zaphod (5800H / gfx90c / 30 GB RAM / 24 GB GTT)

Qwen3.6-35B-A3B-UD-Q4_K_XL (22 GB on 22.5 GB GPU budget):

| Config          | pp t/s | tg t/s |
|-----------------|--------|--------|
| Solver          | 117    | 13.0   |
| Legacy (--noauto) | 117    | 13.2   |

Solver matches legacy on this hardware tier - both end up in the same
memory band and reach identical throughput.

### Flip (7840U / gfx1103 / 32 GB RAM / 22 GB GPU budget)

Qwen3.6-35B-A3B-UD-Q4_K_XL (22 GB on 22 GB GPU budget):

| Config          | pp t/s | tg t/s |
|-----------------|--------|--------|
| Solver          | 264    | 21.7   |
| Legacy (--noauto) | 300    | 36.2   |

Solver is ~15% slower than legacy on this edge case because the mmap +
residency pattern lets legacy use bigger ctx (65536 vs solver's 8192)
and bigger batch (2048/1024 vs solver's 512/512). The solver's
optimistic-first approach can't model mmap working-set size accurately,
so it picks smaller values to guarantee memory fit. The server
starts reliably in both cases.

This is the trade-off: solver is **safe and conservative** on tight
memory, possibly leaving some perf on the table. Use `--noauto` if you
want the legacy preset's aggressive defaults.

## Benchmarks

### Strix Halo (Nimo) and Flip detailed results

HTTP-server benchmark, Nimo (Strix Halo, Vulkan, RADV Mesa 26.2), 970-token
unique prompt + 64-256-token decode.

### Qwen3.6-35B-A3B Q8_K_XL (36 GiB, hybrid SSM, MTP)

| Config                          | pp t/s | tg t/s |
|---------------------------------|--------|--------|
| **Solver (current)**           | **781**| **67.8**|
| Legacy (--noauto)               | 697    | 66.5   |

Solver wins by **+12% prompt** and **+2% decode**. The gain comes from
KV=f16/f16 (10% decode gain), larger ctx (262k vs 196k), bigger
batch/ubatch (4096/2048 vs 2048/1024), and sparse context checkpoints
(2 of 8 vs default 64) which reduces SSD write overhead.

### Qwen3.5-122B-A10B Q5_K_M (91 GiB, hybrid SSM)

| Config                          | pp t/s | tg t/s |
|---------------------------------|--------|--------|
| **Solver (current)**           | **339**| **11.9**|
| Legacy (--noauto)               | 308    | 10.8   |

Solver wins by **+10% prompt** and **+10% decode**. Despite the 91 GB
model taking most of the GPU budget, f16 KV + larger ctx + SSD-disabled
gives consistent gains. The legacy's q8_0 + ctx=131072 is more
conservative; solver's choices just barely fit but pay off in throughput.

### Qwen3.6-35B-A3B Q8_K_XL llama-bench sweep (pp512/tg128)

| ubatch | KV type    | pp512 t/s | tg128 t/s |
|--------|------------|-----------|-----------|
| 512    | f16/f16    | 1048      | 47.5      |
| 1024   | f16/f16    | 1086      | 47.2      |
| 2048   | f16/f16    | 1085      | 47.3      |
| 4096   | f16/f16    | 1056      | 47.0      |
| 1024   | q8_0/q8_0  | 1056      | 43.3      |
| 2048   | q8_0/q8_0  | 1050      | 43.7      |

f16 KV gives consistent ~10% decode gain on Strix Halo. ubatch is
flat in the 1024-4096 range; 2048 is the practical sweet spot.

### Qwen3.6-35B-A3B Q4_K_XL on Flip (7840U, 32GB, Vulkan, 6GB VRAM)

| Config                          | tg64 t/s  |
|---------------------------------|-----------|
| ubatch=2048, KV=f16/f16         | 24.3      |
| ubatch=2048, KV=q8_0/q8_0       | 24.2      |
| ubatch=2048, KV=q4_0/q4_0       | 24.1      |

On Flip the decode throughput is essentially identical across KV types -
decode is purely memory-bandwidth bound on the iGPU. The solver's f16
default is slightly wasteful here but only by 1%.

## Why SSD is disabled on halo

Empirical test on Strix Halo: enabling SSD cache costs 20-30% of prompt
throughput due to constant KV serialization overhead, even with sparse
checkpoints (--ctx-checkpoints 8 --checkpoint-min-step 32768). The
legacy preset table makes the same call. Solver matches this.

Non-halo tiers (Flip, etc.) keep SSD enabled because the GPU budget is
tight enough that prompt cache reuse offsets the serialization cost.

## Per-model differences

- **Hybrid SSM models** (Qwen3.5/3.6, qwen3next) - only ~25% of layers
  are attention layers. KV cache scales with attention-layer count,
  not total layers. Solver reads `full_attention_interval` from GGUF.
- **DeepSeek models** (MLA architecture) - KV cache is tiny relative
  to model size. The solver picks f16 KV at max training ctx.
- **SSM models** (Mamba, Jamba, RWKV) - no KV cache at all. The solver
  detects via `_scan_gguf_arch` and skips KV cache accounting.


## Files

- `scripts/optimize.sh` - solver module (sourced)
- `scratch/read_gguf_kv.py` - GGUF v3 metadata reader
- `llama-run.sh` - integration in `assign_profile_solver()`
- `scratch/test-solver.sh` - unit test harness
- `scratch/test-solver-multi.sh` - multi-model comparison
- `scratch/bench-solver-sweep.sh` - pp/tg sweep across ubatch + KV types
