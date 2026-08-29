# llama-ai

Local LLM inference on AMD APU hardware. Built on
[CachyLLama](https://github.com/fewtarius/CachyLLama), our fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) with a persistent on-disk KV
cache, MoE expert residency, Lightning Indexer, and tight
[CLIO](https://github.com/SyntheticAutonomicMind/CLIO) integration. Self-contained
— no system ROCm install needed. Vulkan (RADV) is the default backend for best
stability on RDNA3 iGPUs.

llama-ai exists to make agentic AI development practical on AMD APUs without
network access. No API keys, no per-token costs, no cloud dependency. The primary
target is a Nimo Axis N161 (Strix Halo, Ryzen AI Max+ 395, Radeon 8060S, 128 GB
unified memory). It also runs on the Ayaneo Flip KB (7840U / 780M / 32 GB) and
the Minisforum UM580 (5800H / 16 GB), plus any other AMD APU in the detection map.

Profiles are computed at runtime by an optimistic-first solver
(`scripts/optimize.sh`) that reads GGUF metadata and iteratively detunes until
the configuration fits the GPU budget.

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/fewtarius/llama-ai.git
cd llama-ai

# Build Vulkan backend (default on Linux AMD)
./scripts/rebuild.sh

# Drop a GGUF model in models/, then:
./llama-run.sh --server
# -> http://localhost:9090
```

---

## Using llama-ai

```bash
# List models found in models/
./llama-run.sh --list-models

# Start server (auto-detects model)
./llama-run.sh --server

# Specific model and backend
./llama-run.sh --server gemma-4-26b --backend vulkan

# Download a model
./llama-run.sh --download Qwen3-14B --quant Q4_K_M

# List available backends
./llama-run.sh --list-backends

# Rebuild
./scripts/rebuild.sh              # Vulkan only (default)
./scripts/rebuild.sh --rocm       # ROCm
./scripts/rebuild.sh --both       # Vulkan + ROCm
./scripts/rebuild.sh --rebuild    # Full rebuild from scratch

# Print the solver's chosen profile for a model
./llama-run.sh --print-profile Qwen3.6-35B-A3B
```

### Reasoning model support

Reasoning models — DeepSeek-R1, Qwen3.6, GLM-4.7 — emit thinking blocks before
each response. By default the runner strips these from prior assistant messages
in the conversation history so they don't waste prompt tokens. To preserve them
across turns, pass `--preserve-reasoning`. Use `--reasoning-budget N` to cap
thinking tokens per response (default: 8192).

### Solver and profile selection

`llama-run.sh` runs an **optimistic-first solver** (in `scripts/optimize.sh`).
The solver reads actual GGUF metadata and system memory/GPU budgets to pick
the best configuration -- it is the single configuration path.

**Solver pipeline:**

1. **Read GGUF metadata** via `scripts/read_gguf_kv.py` -- block count,
   `head_count_kv`, `key/value_length`, `full_attention_interval` (for hybrid
   SSM models), `nextn_predict_layers` (for MTP), `expert_count` (for MoE),
   training context. Reads up to 1 MB of the GGUF to catch architecture-
   specific keys like `laguna.expert_count` or `qwen3next.expert_count`.

2. **Start optimistically** with the per-archetype (batch, ubatch) defaults
   derived from llama-bench sweeps on 17 model/hardware combinations:

   | Tier | Archetype | Size | ubatch | batch |
   |------|-----------|------|--------|-------|
   | halo | dense | any | 1024 | 4096 |
   | halo | MoE small | <60 GB | 1024 | 2048 |
   | halo | MoE large | 60-100 GB | 2048 | 8192 |
   | halo | MoE huge | >=100 GB | 4096 | 8192 |
   | halo | SSM / hybrid | any | 1024 | 4096 |
   | halo | qwen4exp | any | 2048 | 4096 |
   | halo | MLA (DeepSeek-V2/V3/V4, GLM-4.x) | any | 4096 | 8192 |
   | standard | dense | any | 1024 | 2048 |
   | standard | MoE | any | 2048 | 2048 |
   | standard | SSM / hybrid | any | 1024 | 2048 |
   | handheld | dense | any | 512 | 1024 |
   | handheld | MoE | any | 1024 | 2048 |

   Plus: f16/f16 KV, training context (or 4x capped), SSD cache on (Halo:
   off), draft model enabled if found, threads = physical cores (batch) /
   half (gen), MoE strategy = `gpu`. **Checkpoints**: auto-scaled to
   context size (base 8 at 65K ctx, +1 per 8K, capped at 16 during pre-fit
   / 32 after phase 2). Min-step = ctx / checkpoints (floor 8K, or 32K
   when SSD is off).

3. **Score and search** -- builds a priority list of all valid combinations:
   - **Strategy**: `gpu` (300), `cpu` (250), `residency` (200)
   - **Context**: 128K (100), 96K (95), 196K (90), 262K (85), 64K (80)
   - **KV type**: f16 (30), q8_0 (20), q4_0 (10)
   - **Draft**: enabled (5), disabled (0)
   - **Batch/ubatch**: optimistic default (6), partial match (4), other (0)

   Score = `strategy*1000 + ctx*10 + kv + draft + batchub`. First combo
   fitting both GPU and system budgets with minimum cache RAM wins. Memory
   pressure is the real decision driver; the scoring is just a tiebreaker.

   **Strategy selection rules:**
   - Non-MoE/dense/SSM models: only `gpu` is valid (no experts to offload or
     evict). `cpu` and `residency` are skipped entirely.
   - MoE model >80% of GPU budget **and** fits in system RAM (8 GiB OS reserve)
     -> `cpu` preferred, `residency` skipped entirely.
   - Otherwise (MoE): `gpu` -> `residency` -> `cpu`.
   - Low-VRAM (<32 GiB GPU budget), MoE: context capped at 64K, f16 KV
     removed (only q8_0/q4_0).

4. **Cache RAM allocation** -- from **system memory leftover**, not GPU
   leftover: total RAM - OS reserve - config system cost, capped at 25% of
   total RAM with 10% headroom, split between prompt cache and SSD (SSD
   capped at 1 GiB). With n_parallel=1 the prompt cache is set to 0 -- the
   in-memory checkpoint ring covers the same use case at zero cost.

5. **Detune safety net** -- if no fit: ctx = 65536, KV = q4_0/q4_0. If
   still no fit, exits with error. Phase 2 fine-grained detunes: reduce
   KV (q8->q4), reduce NGL, reduce SSD RAM, drop draft, drop SSD, reduce
   ubatch -- each step once.

The solver maps its output to a profile name (`halo-moe-large`,
`std-dense-small`, etc.) for `--print-profile` so users have a single,
stable identifier regardless of which combination was picked. See
[SOLVER.md](SOLVER.md) for the full algorithm, benchmark data, and
hardware notes.

**Override precedence** (low -> high):

| Level | Source |
|-------|--------|
| 1 | Built-in defaults (solver start) |
| 2 | System detection (GPU budget, hardware tier) |
| 3 | Solver output |
| 4 | User overrides (env vars / CLI flags) |

**User overrides recognized:**

| Override | CLI Flag | Env Var |
|----------|----------|---------|
| Context size | `--ctx-size` | `USER_CTX_SIZE` |
| KV cache K/V | `--kv-cache-type` | `KV_CACHE_K_OVERRIDE` / `KV_CACHE_V_OVERRIDE` |
| Threads | `--threads` | `LLAMA_THREADS_OVERRIDE` |
| Ubatch | `--ubatch-size` | `OVERRIDE_UBATCH_SIZE` / `MOE_UBATCH_OVERRIDE` |
| Cache RAM | `--cache-ram` | `OVERRIDE_CACHE_RAM` |
| Disable SSD | `--no-ssd-cache` | `_SSD_DISABLE` / `LLAMA_NO_SSD_CACHE` |
| GPU layers | `-ngl` / `--gpu-layers` | `OVERRIDE_NGL` |
| Fit mode | `--fit on` | `OVERRIDE_FIT` |
| Reasoning budget | `--reasoning-budget` | `OVERRIDE_REASONING_BUDGET` |
| Checkpoint every-N-tokens | `--checkpoint-every-n-tokens` | `OVERRIDE_CHECKPOINT_EVERY` |
| Spec draft p-min (DFlash) | (env) | `LLAMA_SPEC_DRAFT_P_MIN_DFLASH` |
| Spec draft p-min (DSpark) | (env) | `LLAMA_SPEC_DRAFT_P_MIN_DSPARK` |
| Spec draft p-min (MTP) | (env) | `LLAMA_SPEC_DRAFT_P_MIN_MTP` |

Note: `LLAMA_SSD_HOT_RAM`, `LLAMA_SSD_WARM_RAM`, `LLAMA_THREADS`,
`LLAMA_KV_CACHE_TYPE_K` are **defaults** (level 2), not overrides. Use the
`*_OVERRIDE` variants to win over the solver.

When the solver reduces NGL (layers offloaded to GPU) to fit the GPU budget,
`llama-run.sh` caps the launched server's `-ngl` at the solver's value so the
reduction actually applies at launch time (not just in profile output).
`OVERRIDE_FIT` sets `SOLVER_NGL=-1` to let `llama-server` auto-fit, which skips
the cap.

### Hardware-specific behavior

**Strix Halo (Nimo) — 128 GB RAM, ~63 GB GPU budget**
- SSD cache **disabled** (serialization overhead reduces prompt throughput 20–30%).
- Large MoE models (DeepSeek-V4-Flash 97 GB) run with **CPU-MoE + `--load-mode none`**:
  model in system RAM, GPU handles attention + KV only. Avoids Vulkan OOM.

**AYANEO Flip — 26 GB RAM, 22 GB GPU budget**
- SSD cache **enabled** but capped ~1 GiB.
- MoE strategy: `gpu` if the model fits, otherwise `residency` or `cpu`
  per the solver's strategy rules (see "Solver pipeline" above).
- Per-archetype defaults: dense ubatch=1024 batch=2048, MoE ubatch=2048
  batch=2048 (5-10% faster than the old 1024/2048).

**Low-RAM systems (<32 GB total)**
- Prompt cache capped at 2048 MiB.
- SSD default: 512 MiB hot + 512 MiB warm.

### Per-model architecture handling

- **Hybrid SSM** (Qwen3.5/3.6, Qwen3Next): reads `full_attention_interval`,
  scales KV cache (~25% attention layers).
- **DeepSeek MLA**: KV cache very small; f16 at full context usually fits.
- **Pure SSM** (Mamba, Jamba, RWKV): no KV cache; solver uses SSM profile.
- **MTP models**: adds 5% model size (capped 0.5–2 GiB) to GPU+system
  estimates for speculative decoding overhead.




---

## Backends

### Vulkan (Linux/AMD) — default

Uses the Mesa RADV driver. Best stability on RDNA3 iGPUs (Phoenix, Hawk Point,
Strix Point) and earlier GCN/RDNA generations. CPU offloading works for models
that don't fit in GPU memory.

CachyLLama's Vulkan backend carries a wide surface of performance work:

- **DeepSeek-V4 Lightning Indexer** — fused op for compressed-KV attention
- **DSV4 hyper-connection fused ops** — replaces softmax-scale-iterate sequences
- **Quantized-KV FA dequant-once** — dequantizes K/V cache once into f16 scratch
  before flash attention (with host-RAM safety gate)
- **APU `nodes_per_submit` auto-lower** — defaults to 8 on UMA to stay under
  amdgpu `lockup_timeout`
- **Subgroup size pinning** — 32-wide on RDNA3 wave64 for coopmat1 FA

### ROCm (Linux/AMD) — optional

Has known issues on RDNA3 — GLM-4.7-Flash and DeepSeek2 MLA models produce zero
generation tokens. Use Vulkan unless you have a specific reason to try ROCm.
ROCm components carry AMD's license and are downloaded by `scripts/rebuild.sh`.

### Metal (macOS)

Apple Silicon and Intel Macs with Metal-capable GPUs. Build with
`./scripts/rebuild.sh` on macOS — it auto-detects the platform and builds the
Metal backend.

---

## Hardware setup

### GPU memory

AMD APUs share system RAM between CPU and GPU. On the Nimo Axis N161 (Strix
Halo, AI Max+ 395), the BIOS currently carves out **512 MiB** of VRAM — the OS
sees ~125 GB RAM, and GTT defaults to ~62.5 GB. GPU-visible total is ~63 GB.
`detect-gpu.sh` recognizes Strix Halo via PCI ID (`1002:1586`) regardless of the
BIOS VRAM allocation.

Use `apply-ttm-kernel-params.sh` to configure GTT for your hardware:

```bash
# Phoenix/Hawk Point: cap firmware VRAM at 6GB, add 18GB GTT
sudo ./scripts/apply-ttm-kernel-params.sh 18

# Strix Halo: raise GTT to 104 GB (the practical ceiling)
sudo ./scripts/apply-ttm-kernel-params.sh 104
```

The script writes `amdgpu.gttsize`, `ttm.pages_limit`, and `ttm.page_pool_size`
to your bootloader config (GRUB or systemd-boot). SteamFork dual-ESP installs
have a redundant `/boot` — the script probes other vfat partitions and edits the
real loader entry in place. Re-run after OS updates.

Verify after reboot:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E "amdgpu|ttm"
```

### RADV APU memory split

RADV on APUs (`has_dedicated_vram=false`) reports only 2/3 of (VRAM + GTT) as
the DEVICE_LOCAL heap and 1/3 as host heap (game-compat heuristic in
`radv_physical_device.c`). `~/.drirc` enables
`radv_enable_unified_heap_on_apu` for `llama-server`/`llama-cli`/`llama-bench`
so DEVICE_LOCAL = full VRAM + GTT. Without it, models > 2/3 of GPU-visible memory
crash with `vk::DeviceLostError` at load.

### GPU and CPU detection

`scripts/detect-gpu.sh` identifies your AMD GPU via PCI device ID and sets:

| Variable | Description |
|----------|-------------|
| `LLAMA_IS_STRIX_HALO` | `1` on Strix Halo (`1002:1586`), `0` elsewhere |
| `LLAMA_HARDWARE_TIER` | `halo` / `standard` / `handheld` (back-compat) |
| `LLAMA_APU_VRAM_GB` | VRAM carveout in GB |
| `LLAMA_TOTAL_RAM_GB` | Total system RAM in GB |
| `LLAMA_THREADS` | Optimal thread count |
| `LLAMA_GFX_ARCH` | e.g. `gfx1151` (Strix Halo), `gfx1103` (7840U) |
| `LLAMA_GPU_NAME` | e.g. `Radeon 8060S`, `Radeon 780M` |

The script also detects CPU ISA level and generates cmake flags so ISA extensions
(AVX-512 BF16 on Zen 4, AVX2 on Zen 3) are enabled at build time.

Supported generations: Cezanne, Phoenix, Hawk Point, Strix Point, Strix Halo,
Sephiroth, Rembrandt, Mendocino, Renoir, Lucienne.

Overrides:
```bash
LLAMA_GFX_VERSION_OVERRIDE=11.0.3  # skip GPU detection
LLAMA_IS_STRIX_HALO_OVERRIDE=1     # force halo tier
LLAMA_HARDWARE_TIER_OVERRIDE=halo  # force tier string
```

Or via CLI:
```bash
./llama-run.sh --model ... --hardware-tier halo
./llama-run.sh --model ... --is-strix-halo
./llama-run.sh --model ... --no-strix-halo
```

---

## Features

### Auto-profiling

Models are detected by filename and assigned a profile automatically — no manual
configuration needed. The optimistic-first solver reads GGUF metadata and
computes the best configuration for your hardware. MoE models get checkpoint
strategies and reasoning format support. SSM/Mamba models get context-shift
disabled. Large dense models get optimized batch sizes.

The profile is logged at startup, e.g.:
```
Auto profile (solver): halo-moe-small (36GB, MoE=true, SSM=false, MTP=true, qwen4exp=false)
Solver chose: ctx=131072 KV=f16/f16 ubatch=1024 batch=2048 threads=32/16
```

### Hybrid MoE architectures

Qwen3.6 (both dense and MoE variants), Qwen3-Coder-Next, and Qwen3.5-122B mix
attention layers with recurrent (Mamba-2) layers. Laguna-S-2.1 is a DFlash
target with shared experts and sliding window. CachyLLama handles these
correctly: KV cache shifting, attention-only memory clearing that preserves
recurrent state, and checkpoint overflow prevention — so same-conversation state
is always accepted regardless of size.

### MoE expert tracking

MoE models activate only a subset of experts per token — typically 3-8 out of
128-256. CachyLLama tracks which experts activate through `GET /expert-stats`
and `POST /expert-tracking` endpoints. This is instrumentation for now; future
work will use it to reorder experts for cache locality.

### MoE expert residency (run models larger than RAM)

Standard loading OOMs when a MoE model exceeds physical memory, even though only
1-3% of its weights are touched per token. CachyLLama's residency subsystem
keeps only the active subset paged in via `madvise(MADV_WILLNEED /
MADV_DONTNEED)` and lets cold pages spill to SSD. On the Flip KB (25 GB RAM),
this loads and runs MoE models up to 86 GB.

A per-layer recency+frequency cache scores and evicts cold experts (FlashMoE
showed pure LRU evicts hot experts 34% of the time). A co-activation matrix
persists across sessions at `~/.cachylla/coactivation/{model}.json` and informs
future prewarm.

Enabled in `llama-run.sh` automatically for the MoE profile when the hardware
tier is plugged into AC (heavy SSD I/O drains battery). CLI:
`--moe-expert-residency [--moe-resident-per-layer N] [--moe-prewarm-top-k N]`.
See [CachyLLama docs/moe-expert-residency.md](CachyLLama/docs/moe-expert-residency.md)
for architecture, API, tuning, and limitations.

### User isolation

Multi-tenant deployments get per-user KV cache namespacing, slot affinity
(prefer slots already owned by the same user), and configurable concurrency caps
that return HTTP 429 when exceeded. Pass `llama_user_id` in the request body to
enable.

### CLIO integration

CLIO serializes tool definitions with deterministic JSON key ordering and reuses
conversation state across agentic turns. System prompts, tool descriptions, and
compressed context — the static content sent on every API call — are cached and
persisted to disk so they're available immediately on restart.

### SSD-backed KV cache

The feature that makes all of this practical on APU hardware. Cache hits skip
prompt evaluation entirely for the cached prefix, dropping TTFT from minutes to
seconds.

#### Cache tiers

- **Hot** — Current session, in RAM. Instant restore within the same conversation.
- **Warm** — Previous sessions, same server run. In RAM until memory pressure
  pushes them to cold.
- **Cold** — On disk. Survives server restarts.

Tiers promote and demote automatically based on turn activity. Old checkpoints
are evicted when the ring buffer fills. The cache just works — no manual
management needed.

#### Per-tier policy

SSD caching is **disabled on Strix Halo** (halo tier) — the 63 GB GPU-visible
budget holds the working set comfortably, and SSD writes are pure overhead on
the prefill critical path (3 SSD writes per turn at 62 MiB each = 4s of disk
time on a 15K-token prefill). In-memory checkpoints (`--cache-ram` plus
`--ctx-checkpoints`) handle prefix reuse within a server run.

SSD caching is **enabled on standard and handheld tiers** — cold TTFT drops from
minutes to seconds, and the GPU budget is tight enough that prompt cache reuse
offsets the serialization cost.

The `--no-ssd-cache` flag forces SSD caching off on any tier.

By default nothing caps total disk usage — a long MoE conversation can grow to
hundreds of GiB on its own. Set `--cache-ssd-cold-maxsize` (MiB) to bound the
global cold tier footprint; when the cap is hit, the oldest conversation
directories get evicted first. Pair it with `--cache-ssd-max-cold` for
per-conversation checkpoint count limits.

#### System prompt cache

A global cross-conversation cache stores the system section of any prompt after
first evaluation. On a warm server restart, the cache restores the entire system
prompt in under a second — the difference between instant response and waiting
minutes for the first turn.

The cache lives at `kv-cache/{model-stem}/sys-{hash}.bin`, keyed by the first N
tokens of the prompt. Defaults to 8 entries per model with 30-day expiry.

#### Multi-agent sessions (parallel slots)

`--parallel N` (or `-np N`) configures the server with N independent slots, each
holding its own KV cache. Two agentic sessions using the same server in
parallel no longer block each other — the second agent queues onto slot 1 while
the first is generating on slot 0. The host-memory prompt cache
(`--cache-ram`, `--cache-idle-slots`) becomes useful here because slots can
hold divergent state across each other and the LCP matcher hot-swaps them
into fresh tasks.

Per-slot KV cache memory is `n_ctx * layer_count * head_dim * 2 (K+V) *
dtype_bytes`. On a 128K-context f16 model with 60 layers, that's ~2.6 GiB per
slot. With `-np 4` and SSD disabled, that 10 GiB comes out of the GPU budget
the solver would otherwise allocate to model and cache-ram. The solver
accounts for this automatically — if `-np 2` doesn't fit, it shrinks
`n_ctx` or `cache-ram` rather than failing the launch.

CLI: `--parallel 2` or `LLAMA_PARALLEL=2 ./llama-run.sh --server ...`
(LLAMA_PARALLEL only applies when `--parallel` is not given on the command line).
With a single slot (`-np 1`, the default), the solver disables the host-memory
prompt cache automatically — the cache cannot accumulate state when the only
slot is also the one being saved+loaded, and the in-memory checkpoint ring
covers the same use case at zero cost.

---

## Benchmarks

### Strix Halo (Nimo Axis N161)

Radeon 8060S, Vulkan backend. Speedup is prompt eval speedup: cold `prompt_ms`
divided by warm `prompt_ms` (with warm path restoring the prefix from SSD rather
than re-evaluating it). The `benchmarks/` directory holds per-model server logs
from local benchmark runs.

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS (256x8.4B MoE, Lightning Indexer) | 97.0s | 0.34s | **282x** | 6.3-17.5 |
| MiniMax-M2.7 Q2_K_XL (256x4.9B MoE) | 60.3s | 3.19s | **19x** | 12.0-17.2 |
| Qwen3-235B-A22B Thinking-2507 IQ2_M (235B-A22B MoE) | 100.5s | 1.04s | **96x** | 10.6-13.6 |
| Qwen3.5-122B-A10B Q5_K_M (122B-A10B MoE, Mamba hybrid) | 52.9s | 0.31s | **171x** | 9.5-10.6 |
| Qwen3.5-122B-A10B UD-Q4_K_XL (122B-A10B MoE, Mamba hybrid) | 53.5s | 0.28s | **189x** | 23.5-32.7 |
| gpt-oss-120b Q8_K_XL (120B MoE) | 25.0s | 0.24s | **104x** | 18.2-19.9 |
| Qwen3-Coder-Next Q8_K_XL (512x2.5B MoE, Mamba hybrid) | 48.5s | 0.22s | **221x** | 18.9-19.9 |
| Qwen3.6-35B-A3B Q8_K_XL (35B-A3B MoE, Mamba hybrid) | 20.5s | 0.14s | **149x** | 14.2-17.5 |
| GLM-4.7-Flash Q8_K_XL (64x2.6B MoE, MLA) | 85.5s | 0.18s | **478x** | 13.9-20.3 |
| gemma-4-26B-A4B Q5_K_M (26B-A4B MoE, sliding window) | 16.7s | 0.14s | **123x** | 14.0-14.7 |
| Qwen3.6-27B Q8_K_XL (27B dense, Mamba hybrid) | 85.3s | 0.36s | **238x** | 4.7-4.8 |
| Laguna-S-2.1 Q4_K_XL (256x4.5B MoE, DFlash target) | 50.0s | 0.48s | **104x** | 11.2-16.9 |
| gpt-oss-20b Q6_K_XL (20B MoE) | 11.1s | 0.10s | **112x** | 22.3-23.8 |

### Architecture notes

**MoE vs dense.** Qwen3.6-27B is the only dense model in the table — all 27B
parameters activate per token, trading generation speed (4.7-4.8 t/s) for prompt
eval throughput. MoE models activate 2.6-22B parameters per token, hitting 9-32
t/s. Dense models show the largest speedup numbers from SSD caching (238x)
because cold eval is proportionally slower, but generation speed is the
real-world tradeoff.

**Mamba hybrids.** Qwen3.6 (both dense and MoE variants), Qwen3-Coder-Next, and
Qwen3.5-122B interleave attention layers with recurrent (Mamba-2) layers. These
require attention-only KV cache clearing to preserve recurrent state across
cache restores. Qwen3-235B (qwen3moe) is pure attention MoE — no Mamba. GLM-4.7
uses MLA (multi-head latent attention, no Mamba), Gemma 4 uses interleaved
sliding-window/global attention, and DeepSeek-V4-Flash uses Lightning Indexer.
None of these are Mamba hybrids.

**Quantization.** Most models run at Q8_K_XL (30-82 GB on disk). Qwen3.5-122B-
A10B at Q4_K_XL hits 32.7 t/s (vs 9.5 t/s at Q5_K_M) due to lower memory-bandwidth
pressure on decode. MiniMax-M2.7 at Q2_K_XL pushes the frontier — 230B parameters,
256 experts, 4.9B active per token, running in 63 GB GPU-visible. DeepSeek-V4-
Flash IQ3_XXS sits at the top of the model size chart at ~1.6T parameters (256
experts, 8.4B active).

All warm runs restore the cached prefix from SSD; cold from scratch. Warm TTFT
stays under 1.5s across most models — MiniMax-M2.7 at 3.19s is the outlier (its
300 MiB warm-path SSD payload for a 15K-token q8_0 KV cache costs more on the
SSD restore path than smaller models).

### Ayaneo Flip KB

Radeon 780M (gfx1103, Phoenix), 32 GB unified memory (26 GB OS-visible), 2 TB SSD,
Vulkan/RADV backend, `GGML_VK_NODES_PER_SUBMIT=100`. Tested with `./scripts/benchmark.sh`
on `v1.0.0-rc1` (CachyLLama `a461a1fce`). The `benchmarks/vulkan/` directory holds
per-model server logs and drilldowns.

#### Summary

| Model | Size | Profile | pp 2K (t/s) | pp 8K (t/s) | tg 128 (t/s) | tg 512 (t/s) | Cold->Warm TTFT (15K) | Batched @8 |
|-------|------|---------|------------:|------------:|-------------:|-------------:|---------------------:|-----------:|
| gpt-oss-20b-UD-Q6_K_XL | 11.2 GiB | dense-large | 398.0 | 369.9 | 32.0 | 31.6 | 45.8s -> 101ms (**181x**) | 281.2 t/s |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 21.3 GiB | moe-optimized (+residency) | 325.0 | 303.3 | 25.5 | 25.0 | 60.6s -> 150ms (**140x**) | 239.9 t/s |
| gemma-4-26B-A4B-it-UD-Q5_K_M | 16.8 GiB | — | — | — | 0.0 | 0.0 | failed | — |

**pp N (t/s)** = prompt processing throughput at N tokens (llama-bench, tg=128).  
**tg N (t/s)** = text generation throughput at N tokens (llama-bench, pp=512).  
**Cold->Warm TTFT (15K)** = server-side prompt eval for ~15K tokens, cold vs warm (SSD cache restored).  
**Batched @8** = total throughput with 8 parallel prompts (llama-batched-bench, 2048 pp / 128 tg).

#### llama-bench (pure prefill / pure decode)

| Model | pp 512 | pp 2K | pp 8K | pp 16K | tg 128 | tg 512 |
|-------|-------:|------:|------:|-------:|-------:|-------:|
| gpt-oss-20b-UD-Q6_K_XL | 401.8 | 398.0 | 369.9 | 337.5 | 32.0 | 31.6 |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 328.2 | 325.0 | 303.3 | 280.9 | 25.5 | 25.0 |

*Repetitions: 5. Lower variance at larger pp (1-2% sd).*

#### SSD Cache Performance (15K / 5K / 1K token prompts)

| Model | Cold TTFT | Warm TTFT | Speedup | Warm TTFT (steady) |
|-------|----------:|----------:|--------:|-------------------:|
| gpt-oss-20b large (15K) | 45,834 ms | 101 ms | **181x** | 63 ms |
| gpt-oss-20b medium (5K) | 14,589 ms | 69 ms | **119x** | 55 ms |
| gpt-oss-20b small (1K) | 3,803 ms | 52 ms | **58x** | 48 ms |
| Qwen3.6-35B large (15K) | 60,623 ms | 150 ms | **140x** | 63 ms |
| Qwen3.6-35B medium (5K) | 19,067 ms | 106 ms | **76x** | 55 ms |
| Qwen3.6-35B small (1K) | 4,583 ms | 84 ms | **28x** | 48 ms |

*Warm TTFT (steady) = mean of turns 2-5 after first warm restore. First warm turn includes checkpoint load overhead.*

#### llama-batched-bench (concurrent throughput, 2048 pp / 128 tg)

| Model | 1 prompt | 2 prompts | 4 prompts | 8 prompts |
|-------|---------:|----------:|----------:|----------:|
| gpt-oss-20b-UD-Q6_K_XL | 232.0 t/s | 258.4 t/s | 277.2 t/s | 281.2 t/s |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 185.8 t/s | 215.9 t/s | 233.9 t/s | 239.9 t/s |

*Total aggregate throughput. Per-slot throughput drops with concurrency (281.2/8 = 35 t/s).*

#### Notes

- **gpt-oss-20b** is a dense 20B model — all parameters activate per token. Higher
  generation throughput (32 t/s) and best cache speedup (181x at 15K) because the
  KV cache is smaller and the model fits comfortably in the 22 GiB GPU budget.
  Runs the `dense-large` profile (f16 KV, no SSD on Halo; on Flip SSD is enabled
  with q8_0 KV, ubatch=512, batch=1024, threads=4/8).

- **Qwen3.6-35B-A3B** is a 35B MoE model (3B active, A3B = 3B active experts).
  Runs the `moe-optimized` profile with `--moe-expert-residency` (madvise
  MADV_FREE eviction, 64 resident experts/layer) and `kv-unified`. The residency
  subsystem keeps hot experts paged in so cold-path expert loads don't dominate
  decode. Generation is memory-bandwidth bound (~25 t/s) on this hardware.
  Context capped at 65536 by the solver for MoE on handheld tier.

- **gemma-4-26B-A4B** failed to generate (0 t/s across all tests). The model
  loads but produces no output tokens — likely a tokenizer/chat template mismatch
  or Vulkan kernel issue with the sliding-window attention pattern. Investigation
  needed.

- **GGML_VK_NODES_PER_SUBMIT=100** was used (vs conservative default 8). On the
  7840U/Phoenix this gives ~4.6% higher tg64 throughput (22.65 vs 21.65 t/s) with
  no amdgpu lockup_timeout observed. The conservative default (8) is kept in
  CachyLLama to avoid regressions on other APUs.

- All models use the optimistic-first solver (`scripts/optimize.sh`).
  Profiles are computed from GGUF metadata and the 26 GB OS-visible /
  22 GiB GPU budget.

- Full per-model logs and drilldowns in `benchmarks/vulkan/<model>/`:
  `summary.md` (this table), `cache/analysis.md`, `bench/report.md`,
  `batched/report.md`, `summary.json`.

### Real-world CLIO performance

CLIO sends 18-30K tokens of system prompt, tool definitions, and conversation
context on every API call. Every prompt has a static prefix (~18K tokens --
system prompt, tool definitions) that's identical across all turns, and a
dynamic tail that grows as the conversation progresses.

Four sessions, all given the same task -- "Spend a few minutes studying this
project and then explain it" (read files, check git history, run commands, write
a final analysis). Two ran on the Nimo Axis N161 (Strix Halo, Ryzen AI Max+ 395,
128 GB unified memory, 4 TB NVMe RAID) and two on the Ayaneo Flip KB (784U, 32 GB
unified memory, 2 TB SSD). Qwen3.6 uses self-MTP speculative decoding
(`--spec-type draft-mtp`, n_max=2); DeepSeek and Laguna do not use spec decoding
(Laguna uses DFlash as a target architecture). All sessions use flash attention
and reasoning budget 8192.

Metrics are from `scripts/log_analyzer.py` on the paired server logs in
`scratch/`. PP and Gen are weighted by token count (total prompt tokens divided
by total PP time; total decode tokens divided by total decode time).

#### Session summary

| # | Hardware | Model | Context | Draft | KV | Batch/Ubatch | Threads | GPU strat. | SSD | Cache-RAM | Turns | Time | PP (t/s) | Gen (t/s) | Draft acc. | LCP sim. | f_keep |
|---|----------|-------|---------|-------|----|:---------:|---------|----------|-----|-----------|-------|------|---------:|----------:|------------|:--------:|:------:|
| 1 | Nimo (Strix Halo) | Qwen3.6-35B-A3B Q8_K_XL | 131072 | Yes (self-MTP) | f16/f16 | 4096/2048 | 16/32 | gpu | off | 31 GB | 3 | 2:24 | 68 | 47 | 84% | 0.71 | 0.99 |
| 2 | Flip KB (784U) | Qwen3.6-35B-A3B Q4_K_XL | 65536 | Yes (self-MTP) | q8_0/q8_0 | 1024/512 | 4/8 | residency | 124 MiB | 5 GB | 6 | 5:19 | 22 | 21 | 85% | 0.93 | 0.99 |
| 3 | Nimo (Strix Halo) | DeepSeek-V4-Flash IQ3_XXS | 131072 | No | q4_0/q4_0 | 4096/2048 | 16/32 | gpu | off | 5 GB | 4 | 6:03 | 184 | 12 | -- | 0.90 | 1.00 |
| 4 | Nimo (Strix Halo) | Laguna-S-2.1 Q5_K_XL | 131072 | No (DFlash) | q8_0/q8_0 | 4096/2048 | 16/32 | gpu | off | 11 GB | 16 | 9:36 | 376 | 16 | -- | 0.90 | 1.00 |

First turn evaluates the full system prompt cold. Every subsequent turn uses
in-memory checkpoints to restore the cached prefix, evaluating only the new
content. Draft acceptance is the per-turn average; overall acceptance across all
generated tokens is 72.5% (Nimo) and 76.8% (Flip). **LCP sim.** = Longest Common Prefix similarity score (mean across warm turns, threshold 0.20); **f_keep** = fraction of prompt tokens restored from checkpoint (prompt cache hit rate).

#### Session 1 -- Qwen3.6-35B-A3B Q8_K_XL (Nimo, Strix Halo)

Qwen3.6-35B-A3B-UD-Q8_K_XL, 131072 context, self-MTP draft (n_max=2), f16/f16 KV,
flash attention, 16/32 threads, batch/ubatch 4096/2048, GPU strategy, SSD off,
31 GB cache-RAM. GPU budget: ~61 GiB. LCP sim: 0.948, 0.471; f_keep: 0.993, 0.986.

| Turn | Prompt tokens | Decode tokens | TTFT (s) | PP (t/s) | Gen (t/s) |
|---|--:|--------------:|--------------:|---------:|---------:|
| T0 | 24,668 | 222 | 31.8 | 775.6 | 51.7 |
| T87 | 1,410 | 427 | 1.9 | 741.5 | 56.1 |
| T240 | 29,417 | 1,312 | 47.9 | 613.5 | 43.4 |

Total: 2:24 (55,495 prompt + 1,961 decode = 57,456 tokens). Draft acceptance:
94.2%, 92.3%, 64.4%.

#### Session 2 -- Qwen3.6-35B-A3B Q4_K_XL (Flip KB)

Qwen3.6-35B-A3B-UD-Q4_K_XL, 65536 context, self-MTP draft (n_max=2), q8_0/q8_0 KV,
flash attention, 4/8 threads, batch/ubatch 1024/512, MoE residency strategy,
SSD on (124 MiB), 5 GB cache-RAM, `--moe-expert-residency` + `--load-mode mmap`.
GPU budget: 22 GiB. LCP sim: 0.969, 0.888, 0.880, 0.946, 0.971; f_keep: 0.995, 0.993, 0.994, 0.991, 0.994.

| Turn | Prompt tokens | Decode tokens | TTFT (s) | PP (t/s) | Gen (t/s) |
|---|--:|--------------:|--------------:|---------:|---------:|
| T0 | 24,432 | 182 | 109.1 | 224.0 | 24.5 |
| T89 | 843 | 237 | 5.1 | 164.2 | 24.4 |
| T176 | 3,251 | 213 | 19.0 | 171.0 | 22.7 |
| T258 | 3,954 | 303 | 23.4 | 168.9 | 20.9 |
| T384 | 1,840 | 214 | 11.6 | 158.3 | 21.7 |
| T467 | 1,009 | 1,274 | 6.6 | 152.6 | 20.5 |

Total: 5:19 (35,329 prompt + 2,423 decode = 37,752 tokens). Draft acceptance:
96.0%, 92.2%, 90.8%, 76.9%, 87.8%, 68.5%.

#### Session 3 -- DeepSeek-V4-Flash IQ3_XXS (Nimo, Strix Halo)

DeepSeek-V4-Flash-731-UD-IQ3_XXS, 131072 context, no draft, q4_0/q4_0 KV, flash
attention, 16/32 threads, batch/ubatch 4096/2048, GPU strategy, SSD off, 5 GB
cache-RAM. GPU budget: ~61 GiB. LCP sim: 0.964, 0.824, 0.917; f_keep: 0.997, 0.996, 0.995.

| Turn | Prompt tokens | Decode tokens | TTFT (s) | PP (t/s) | Gen (t/s) |
|---|--:|--------------:|--------------:|---------:|---------:|
| T0 | 24,786 | 171 | 115.0 | 215.5 | 11.8 |
| T178 | 1,023 | 170 | 10.0 | 102.1 | 11.9 |
| T349 | 5,607 | 238 | 39.0 | 143.9 | 11.8 |
| T589 | 2,934 | 1,332 | 22.8 | 128.6 | 11.7 |

Total: 6:03 (34,350 prompt + 1,911 decode = 36,261 tokens).

#### Session 4 -- Laguna-S-2.1 Q5_K_XL (Nimo, Strix Halo)

Laguna-S-2.1-UD-Q5_K_XL, 131072 context, no spec-decode draft (DFlash target
architecture), q8_0/q8_0 KV, flash attention, 16/32 threads, batch/ubatch
4096/2048, GPU strategy, SSD off, 11 GB cache-RAM. GPU budget: ~61 GiB. LCP sim: 0.998, 0.957, 0.707, 0.979, 0.999, 0.753, 0.920, 0.852; f_keep: 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.996.

| Turn | Prompt tokens | Decode tokens | TTFT (s) | PP (t/s) | Gen (t/s) |
|---|--:|--------------:|--------------:|---------:|---------:|
| T0 | 24,113 | 225 | 48.0 | 502.7 | 17.2 |
| T233 | 54 | 102 | 0.9 | 63.4 | 17.5 |
| T337 | 1,093 | 136 | 3.0 | 359.1 | 17.4 |
| T475 | 10,634 | 206 | 26.0 | 408.7 | 16.8 |
| T686 | 773 | 165 | 2.7 | 285.6 | 16.8 |
| T853 | 54 | 189 | 0.9 | 62.8 | 16.8 |
| T1044 | 12,403 | 198 | 33.6 | 368.9 | 16.2 |
| T1247 | 4,364 | 224 | 13.2 | 329.4 | 16.0 |
| T1474 | 9,519 | 298 | 28.8 | 330.2 | 15.6 |
| T1776 | 2,472 | 389 | 8.7 | 283.4 | 15.4 |
| T2168 | 1,658 | 225 | 5.7 | 289.8 | 15.4 |
| T2395 | 2,032 | 286 | 6.9 | 293.5 | 15.3 |
| T2683 | 2,323 | 377 | 8.7 | 265.6 | 15.2 |
| T3063 | 1,008 | 155 | 4.0 | 250.4 | 15.2 |
| T3220 | 63 | 51 | 1.0 | 60.2 | 15.3 |
| T3273 | 63 | 2,114 | 1.1 | 59.9 | 15.1 |

Total: 9:36 (72,626 prompt + 5,340 decode = 77,966 tokens).

#### Key observations

- **Strix Halo dominates.** Session 1 (Qwen3.6 Q8_K_XL, halo) hits 68 t/s prompt
  and 47 t/s generation, versus 22/21 t/s on Flip (Session 2) with the same model
  at Q4_K_XL. The ~61 GiB GPU budget and f16 KV cache let the Lightning Indexer
  run at full throughput.
- **DeepSeek IQ3_XXS is compute-bound.** At 184 t/s prompt and 12 t/s generation,
  the 1.6T-parameter model pushes the GPU hard -- generation is 3-4x slower than
  Qwen3.6 despite similar parameter count, because DeepSeek's compressed-KV MLA
  architecture has higher per-token compute.
- **Laguna has the highest token throughput.** With 376 t/s prompt and 16 t/s
  generation over 16 turns and 77,966 tokens, it sustains steady performance
  across a long conversation. The DFlash target architecture and q8_0 KV keep
  memory pressure low on the halo budget.
- **Flip's small budget limits batch/ubatch.** The 22 GiB GPU budget forces
  batch/ubatch down to 1024/512 (from 4096/2048 on halo), plus MoE residency
  and SSD caching. Despite this, generation (21 t/s) is close to half of halo's
  (47 t/s), not the 4x you might expect from the GPU gap -- the bottleneck shifts
  to decode rather than prefill.
- **Draft model helps Qwen3.6.** The self-MTP draft (n_max=2) delivers 84-85%
  acceptance, cutting generated tokens roughly in half. DeepSeek and Laguna run
  without spec decoding.

---

## Repo structure

```
llama-ai/
├── llama-run.sh              # Main entry point — model detection, server launch
├── CachyLLama/               # Submodule — our fork of llama.cpp
├── scripts/
│   ├── rebuild.sh            # Build script (Vulkan default, optional ROCm)
│   ├── env.sh                # Environment setup (source before ROCm tools)
│   ├── detect-gpu.sh         # GPU/APU and CPU ISA auto-detection
│   ├── optimize.sh           # Optimistic-first solver (sourced by llama-run.sh)
│   ├── benchmark.sh          # KV cache performance testing
│   ├── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
│   ├── install-deps.sh       # Dependency installer
│   ├── lib-discover-models.sh  # HuggingFace model discovery
│   ├── read_gguf_kv.py       # GGUF metadata reader
│   └── log_analyzer.py       # Benchmark log analysis
├── src/
│   ├── cachy-llama-vulkan/   # Vulkan build output + build.sh
│   ├── cachy-llama-rocm/     # ROCm build output + build.sh
│   └── cachy-llama-metal/    # Metal build output + build.sh (macOS)
├── deps/                     # ROCm SDK (downloaded by rebuild.sh, gitignored)
├── models/                   # GGUF files (gitignored)
├── kv-cache/                 # SSD-backed KV cache (gitignored)
├── scratch/                  # Transient working files (gitignored)
└── benchmarks/               # Results with full server logs
```

See [AGENTS.md](AGENTS.md) for the technical reference (build commands, code
style, directory conventions).

---

## License

**Source code:** [GPL-3.0-or-later](LICENSE)
**Documentation:** [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

CachyLLama (the inference engine) is MIT-licensed (same as upstream llama.cpp,
Copyright (c) 2023-2026 The ggml authors). The llama-ai runner scripts, GPU
detection, solver, and benchmark harness are GPL-3.0-or-later. ROCm components
carry AMD's license. See the [CachyLLama LICENSE](CachyLLama/LICENSE) and
[CachyLLama README](CachyLLama/README.md#license) for details.
