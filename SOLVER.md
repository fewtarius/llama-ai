# Solver Reference (llama-ai)

The solver in `scripts/optimize.sh` is the single configuration path for
`llama-run.sh`. It replaces the old static preset table and uses an
optimistic-first, memory‑aware search to pick the best settings for the
model and hardware.

## Why

Static presets cannot adapt to the wide variety of model architectures,
quantizations, and unified‑memory APU configurations. The solver reads
actual GGUF metadata and system memory/GPU budgets to:

- Use **f16 KV** when it fits (faster decode on big GPUs).
- Use **q8_0 / q4_0** only when memory is tight.
- Choose the **correct MoE strategy**:
    - `gpu` when the whole model fits comfortably.
    - `cpu` (CPU‑MoE, `--load-mode none`) when the model is large and fits
      in system RAM; this avoids the Vulkan OOM that `residency` can cause
      on unified‑memory APUs.
    - `residency` only as a fallback when neither `gpu` nor `cpu` is safe.
- Allocate **cache RAM** from actual leftover system memory, never
  starving the OS or the GPU.
- **Fail fast** when no workable configuration exists instead of
  silently degrading to an undersized preset.

## How

The solver proceeds in five stages:

### 1. Read GGUF Metadata

Uses `scripts/read_gguf_kv.py` to extract:

- `block_count`, `embedding_length`, `head_count`, `head_count_kv`
- `key_length`, `value_length`
- `full_attention_interval`
- `expert_count`, `nextn_predict_layers` (for MoE / MTP detection)
- `qwen4exp.*` arch keys (Qwen3.8-Flash-Next: PLE, hyper-connection, QSA)
- `context_length`

This data drives all memory and performance calculations.

### 2. Start Optimistically

- KV cache: **f16/f16**
- Context: training context (or 4× training, capped)
- SSD cache: enabled (Halo: disabled)
- Draft model: enabled if found (DSpark / DFlash / MTP)
- Ubatch: 2048 (Halo), 1024 (standard), 512 (handheld)
- Threads: batch = physical cores, gen = half of that
- MoE strategy: `gpu` (all layers offloaded)
- Checkpoints: auto-scaled to context size (base 8 at 65K ctx, +1 per 8K,
  capped at 16 during pre-fit / 32 after phase 2). Min-step = ctx /
  checkpoints (floor 8K, or 32K when SSD is off).

### 3. Model Memory

The solver calculates both **GPU memory** and **system memory** for a
candidate. Key components:

- **Model GPU footprint**
    - `gpu`: 100% of model
    - `cpu`: ~6% (attention/embedding only)
    - `residency`: ~30% (attention + small expert cache)
- **KV cache** per token from GGUF dimensions
- **Draft model + draft KV**
- **MTP overhead** (5% of model, capped 0.5–2 GiB)
- **Compute buffers** (512 MiB + ubatch × 256 B)
- **System reserve** (256 MiB)
- **SSD hot/warm RAM** (if enabled)

The GPU budget is computed in `llama-run.sh` as:

    min(VRAM + GTT, total_system_RAM - OS_reserve)

This is crucial for unified‑memory APUs; it prevents overcommitting
GPU‑accessible memory beyond what the OS can spare.

### 4. Scored Search (Phase 1)

The solver builds a priority list of all valid combinations of:

- Strategy: `gpu` (score 300), `cpu` (score 250), `residency` (score 200)
- Context: 128K (100), 96K (95), 196K (90), 262K (85), 64K (80)
- KV: f16 (30), q8_0 (20), q4_0 (10)
- Draft: enabled (5), disabled (0)

Final score = `strategy*1000 + ctx*10 + kv + draft`.

The first combination that fits both GPU and system memory budgets and
leaves at least the minimum required cache RAM wins.

**Strategy selection rules**:

- Non-MoE/dense/SSM models: only `gpu` is valid (no experts to offload or
  evict). `cpu` and `residency` are skipped entirely.
- If the model is **MoE** and **model size > 80% of GPU budget** and
  the model fits in system RAM with an 8 GiB OS reserve, `cpu` is
  preferred over `residency`. `residency` is skipped entirely.
- Otherwise (MoE), the order is `gpu -> residency -> cpu`.
- On low‑VRAM systems (<32 GiB GPU budget), MoE models:
    - Context is capped at 64K.
    - f16 KV is removed; only q8_0 / q4_0 are considered.

### 5. Cache RAM Allocation

After a config is chosen, the solver computes **cache RAM from system
memory leftover**, not GPU leftover:

- Calculate system memory needed by the chosen config (excluding caches).
- Subtract from `total_system_RAM - OS_reserve`.
- Take the smaller of this leftover and the GPU leftover.
- Apply 10% headroom, then cap at 25% of total RAM.
- Split between prompt cache (`--cache-ram`) and SSD cache (if enabled),
  with SSD capped at 1 GiB.

The prompt cache size is stored in `SOLVER_CACHE_RAM`; the SSD hot/warm
sizes are stored in `SOLVER_SSD_HOT_RAM` and `SOLVER_SSD_WARM_RAM`.

## Detune Safety Net

If Phase 1 finds no fit, the solver falls back to the absolute minimum:

    ctx = MIN_CTX (default 65536)
    KV = q4_0/q4_0

If even that does not fit, `llama-run.sh` prints an error and exits.

Phase 2 performs fine‑grained detunes if the chosen config still
exceeds budget: reduce KV (q8→q4), reduce NGL, reduce SSD RAM,
drop draft, drop SSD, reduce ubatch. Each step is applied only once.

## Override Precedence

Lowest to highest:

1. Built‑in defaults (solver starts here)
2. System detection (GPU budget, hardware tier)
3. Solver output
4. User overrides via env vars and CLI flags

User overrides recognized by `apply_user_overrides()`:

| Override                  | Source                        |
|---------------------------|-------------------------------|
| `USER_CTX_SIZE`           | `--ctx-size`                  |
| `KV_CACHE_K_OVERRIDE`     | `--kv-cache-type`             |
| `KV_CACHE_V_OVERRIDE`     | `--kv-cache-type`             |
| `LLAMA_THREADS_OVERRIDE`  | `--threads`                   |
| `MOE_UBATCH_OVERRIDE`     | env                           |
| `OVERRIDE_UBATCH_SIZE`    | `--ubatch-size`               |
| `OVERRIDE_CACHE_RAM`      | `--cache-ram`                 |
| `_SSD_DISABLE`            | `--no-ssd-cache`              |
| `OVERRIDE_NGL`            | `-ngl` / `--gpu-layers`       |
| `OVERRIDE_FIT`            | `--fit on`                    |
| `OVERRIDE_REASONING_BUDGET`| `--reasoning-budget`          |

Note: `LLAMA_SSD_HOT_RAM`, `LLAMA_SSD_WARM_RAM`, `LLAMA_THREADS`, and
`LLAMA_KV_CACHE_TYPE_K` are **defaults**, not overrides. To override the
solver, use the corresponding `*_OVERRIDE` variable or CLI flag.

## Hardware Notes

### Strix Halo (Nimo) – 128 GB RAM, ~63 GB GPU budget

- SSD cache is **disabled** because serialization overhead reduces prompt
  throughput by 20–30%.
- Large MoE models (e.g., DeepSeek‑V4‑Flash 97 GB) are run with
  **CPU‑MoE + `--load-mode none`**. The model sits in system RAM; the
  GPU handles only attention and KV. This avoids Vulkan OOM and
  utilises the full 128 GB RAM.

### AYANEO Flip – 26 GB RAM, 22 GB GPU budget

- SSD cache is **enabled** but capped to ~1 GiB.
- MoE models are run with **residency** or **CPU–MoE** if small enough.
  Non-MoE dense models use the `gpu` strategy only.
- Context is limited to 64K for MoE models; KV is q8_0 or q4_0.

### Low‑RAM systems (<32 GB total RAM)

- Prompt cache is capped at 2048 MiB.
- OS reserve for handheld tier is 4 GiB (not 8 GiB); for standard/halo it
  is 8 GiB. The reserve protects OS responsiveness under GPU memory pressure.
- SSD cache default: 512 MiB hot + 512 MiB warm.

## Per‑Model Architecture Handling

- **Hybrid SSM models** (Qwen3.5/3.6, Qwen3Next):
  Only ~25% of layers are attention layers. The solver reads
  `full_attention_interval` and scales KV cache accordingly.
- **DeepSeek MLA models**:
  KV cache is very small; f16 at full context usually fits.
- **Pure SSM models** (Mamba, Jamba, RWKV):
  No KV cache. The solver skips KV accounting and uses the SSM profile.
- **MTP models**:
  The solver adds 5% model size (capped 0.5–2 GiB) to both GPU and
  system memory estimates to cover speculative decoding overhead.
- **Qwen3.8-Flash-Next (qwen4exp)**:
  Has a PLE n-gram embedding (~40% of model) that is always on GPU and
  not offloadable by reducing -ngl. The solver subtracts it from the GPU
  footprint in `_opt_model_gpu_footprint` and `llama-run.sh` adds
  `-ot per_layer_token_embd.weight=CPU` to offload the PLE to CPU,
  keeping all MoE computation on GPU (2x faster decode than --cpu-moe).
  `full_attention_interval=4` means only 12/48 layers store KV cache
  (GDN layers use linear attention). f16 KV is required — quantized KV
  crashes the QSA assert. MTP is enabled only when the GGUF contains
  `nextn_predict_layers` (Unsloth quants typically don't include them).

## Files

- `scripts/optimize.sh` – solver module (sourced)
- `scripts/read_gguf_kv.py` – GGUF v3 metadata reader
- `llama-run.sh` – integration in `assign_profile_solver()`
- `scratch/test-solver.sh` – unit test harness
- `scratch/test-solver-multi.sh` – multi-model comparison
- `scratch/bench-solver-sweep.sh` – pp/tg sweep across ubatch + KV types

