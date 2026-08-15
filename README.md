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
the configuration fits the GPU budget. A legacy preset table is preserved as a
fallback (`--noauto`).

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
thinking tokens per response (default: 2048).

### Solver and profile selection

`llama-run.sh` runs an optimistic-first solver by default. The solver:

1. Reads GGUF metadata (block count, head_count_kv, key/value_length,
   full_attention_interval for hybrid SSM models, nextn_predict_layers for MTP)
2. Starts with the maximum-performance configuration (f16/f16 KV, max context,
   SSD cache on halo, draft model enabled)
3. Computes the memory cost (model + KV cache + draft + SSD buffers + compute)
4. Iteratively detunes by least performance impact until the config fits the
   GPU budget

**Detune priority:** SSD RAM → context size → ubatch → threads → V-cache q8_0 →
V-cache q4_0 → K-cache q8_0 → drop draft → drop SSD → MoE residency →
MoE cpu-moe → auto layer split.

User overrides always win: `--ctx-size`, `--ubatch-size`, `--cache-ram`,
`--threads`, `--no-ssd-cache`, `--hardware-tier`, and all `*_OVERRIDE` env vars.

Pass `--noauto` to use the legacy preset table instead:

| Preset | When | ctx | KV | batch | SSD | cpu-moe |
|--------|------|-----|-----|-------|-----|---------|
| `halo-moe-large` | Strix Halo, MoE, >50 GB | 131072 | q8_0 | 2048/512 | off | — |
| `halo-moe-small` | Strix Halo, MoE, ≤50 GB | 196608 | f16 | 2048/512 | off | — |
| `halo-dense` | Strix Halo, dense (any) | 131072 | f16 | 2048/512 | off | — |
| `std-moe-large` | non-Halo, MoE, >18 GB | 65536 | q8_0 | 1024/256 | on | auto¹ |
| `std-moe-small` | non-Halo, MoE, ≤18 GB | 32768 | q8_0 | 1024/256 | on | auto¹ |
| `std-dense-large` | non-Halo, dense, >15 GB | 32768 | q4_0 | 1024/256 | on | — |
| `std-dense-small` | non-Halo, dense, ≤15 GB | 32768 | q8_0 | 1024/256 | on | — |
| `ssm` | Mamba / Jamba / RWKV | 65536 (262144 on halo) | q8_0 | 1024/512 | off | — |

¹ `--cpu-moe + --load-mode none` is added automatically when the model is ≥23 GiB
to avoid the mmap + `--moe-expert-residency` page-fault pathology. Halo tier
never triggers this because GTT holds the model.

The solver maps its output to a legacy profile name for `--print-profile` so
existing scripts keep working. See [SOLVER.md](SOLVER.md) for the full
algorithm, override precedence, and benchmark data.

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
Auto profile (solver): std-moe-large (35GB, MoE=true, SSM=true)
Solver chose: ctx=262144 KV=f16/f16 ubatch=4096 batch=4096 threads=8/16
Solver detune steps: reduce SSD RAM, increase context, enable f16 KV
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

Radeon 780M, 32 GB unified memory, Vulkan backend.

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| gemma-4-26B Q5_K_M (26B, dense) | 63.8s | 0.46s | **138.4x** | 14-17 |
| gpt-oss-20b Q6_K_XL (20B, 3.6B active MoE) | 45.9s | 0.24s | **193.5x** | 25-31 |
| Qwen3.6-35B-A3B Q4_K_XL (35B, 3B active MoE) | 70.9s | 0.38s | **187.9x** | 19-24 |

Gemma 4 generation nearly tripled compared to the previous Flip benchmark
(6 -> 14-17 t/s), with cold TTFT also improving 33% (94.8s -> 63.8s).
Qwen3.6-35B-A3B cold eval improved 16% (84s -> 70.9s) while generation maintained
and extended its range (19-24 t/s).

MoE expert residency keeps hot experts paged in via `madvise(MADV_WILLNEED)`, so
cold-path expert loads no longer dominate decode. All warm TTFTs converge under
500ms — the SSD cache bridges the hardware gap between Flip KB and Strix Halo on
subsequent turns.

### Real-world CLIO performance (Strix Halo)

CLIO sends 18-30K tokens of system prompt, tool definitions, and conversation
context on every API call. Every prompt has a static prefix (~18K tokens — system
prompt, tool definitions) that's identical across all turns, and a dynamic tail
that grows as the conversation progresses.

This test has CLIO evaluate a project: read files, check git history, run
commands, write a final analysis. Qwen3.6-35B Q8_K_XL, 196K context, 32 threads,
Vulkan backend.

| Turn | Prompt tokens | TTFT | Gen t/s |
|------|---------------|------|---------|
| T0 | 21,336 | 26.5s | 36.5 |
| T1 | 1,623 | 2.8s | 37.9 |
| T2 | 5,152 | 8.0s | 37.7 |
| T3 | 4,428 | 7.5s | 38.2 |

**Total: 88 seconds** (32,539 prompt tokens, 1,656 generated). First turn
evaluates the full system prompt cold. Every subsequent turn uses in-memory
checkpoints to restore the cached prefix, evaluating only the new content.
Generation holds steady at 36-38 t/s across all turns.

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
