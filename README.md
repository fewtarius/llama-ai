# llama-ai

Local LLM inference on AMD APU hardware. Built on
[CachyLLama](https://github.com/fewtarius/CachyLLama), our fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) with an SSD-backed KV
cache, agentic workflow tuning, and tight
[CLIO](https://github.com/SyntheticAutonomicMind/CLIO) integration.
Self-contained - no system ROCm install needed. Vulkan (RADV) is the
default backend for best stability on RDNA3 iGPUs.

llama-ai exists to make agentic AI development practical on AMD APUs
without network access. No API keys, no per-token costs, no cloud
dependency. The primary target is a Nimo Axis N161 (Ryzen AI Max+ 395,
Radeon 8060S, 128GB unified memory with 96GB pre-allocated to the APU).
It also runs on the Ayaneo Flip KB (7840U / Radeon 780M / 32GB) and
the Minisforum UM580 (5800H / 16GB), plus any other AMD APU in the
detection map.

Profiles scale automatically with the APU's VRAM carveout via
`LLAMA_HARDWARE_TIER` (`handheld`, `standard`, `halo`): halo gets
192K-token context on MoE models and 196K on GPT-OSS, handheld keeps
conservative 64K-token settings tuned for the 780M's 6GB VRAM envelope.

Performance work on hybrid architectures - Qwen3.6, Qwen3-Coder-Next,
Qwen3.5-122B (all attention+Mamba MoE), and Laguna-S-2.1 (DFlash target)
- lives as commits in the
[CachyLLama](https://github.com/fewtarius/CachyLLama) git history. DeepSeek-V4-Flash
uses Lightning Indexer (deepseek4) plus MLAttention. GLM-4.7 uses MLA
(multi-head latent attention, deepseek2) and Gemma 4 uses
sliding-window/global attention; neither is a Mamba hybrid. The submodule
here points at the fork, not at ggml-org/llama.cpp.

## Quick start

```bash
git clone --recurse-submodules https://github.com/fewtarius/llama-ai.git
cd llama-ai

# Build Vulkan backend (default)
./scripts/rebuild.sh

# Drop a GGUF model in models/, then:
./llama-run.sh --server
# -> http://localhost:9090
```

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
```

### Reasoning model support

Reasoning models - DeepSeek-R1, Qwen3.6, GLM-4.7 - emit thinking
blocks before each response. By default the runner strips these from
prior assistant messages in the conversation history so they don't
waste prompt tokens. To preserve them across turns, pass
`--preserve-reasoning`. Use `--reasoning-budget N` to cap thinking
tokens per response (default: 2048).

## Backends

### Vulkan (Linux/AMD)

Default. Uses the Mesa RADV driver. Best stability on RDNA3 iGPUs
(Phoenix, Hawk Point, Strix Point) and earlier GCN/RDNA generations.
CPU offloading works for models that don't fit in GPU memory.

### ROCm (Linux/AMD)

Optional. Has known issues on RDNA3 - GLM-4.7-Flash and DeepSeek2 MLA
models produce zero generation tokens. Use Vulkan unless you have a
specific reason to try ROCm.

### Metal (macOS)

Apple Silicon and Intel Macs with Metal-capable GPUs. Build with
`./scripts/rebuild.sh` on macOS - it auto-detects the platform and
builds the Metal backend.

## Hardware setup

### GPU memory

AMD APUs share system RAM between CPU and GPU. Use
`apply-ttm-kernel-params.sh` to configure GTT:

```bash
# Phoenix/Hawk Point: cap firmware VRAM at 6GB, add 18GB GTT
sudo ./scripts/apply-ttm-kernel-params.sh 18

# Strix Halo with 96GB BIOS carveout: no GTT changes needed
sudo ./scripts/apply-ttm-kernel-params.sh
```

The script writes `amdgpu.gttsize`, `amdgpu.vis_vramlimit`, and
`ttm.pages_limit` to your bootloader config. Supports GRUB and
systemd-boot. Defaults are tier-aware:

| Tier     | Examples                      | vis_vramlimit | GTT      |
|----------|-------------------------------|---------------|----------|
| handheld | 780M, 890M (Phoenix/Hawk)     | 6GB           | RAM-6GB  |
| standard | 16-32GB APU VRAM              | 16GB          | 8GB      |
| halo     | 8060S (Strix Halo)            | not set       | 4GB      |

The `halo` tier skips `vis_vramlimit` to preserve the BIOS VRAM
carveout (typically 96GB). Override with `VIS_VRAM_LIMIT_MB` or the
first positional argument. Verify after reboot:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E "amdgpu|ttm"
```

### GPU and CPU detection

`detect-gpu.sh` identifies your AMD GPU via PCI device ID and sets the
right environment for both Vulkan and ROCm. Supported generations:
Cezanne, Phoenix, Hawk Point, Strix Point, Strix Halo, Sephiroth,
Rembrandt, Mendocino, Renoir, Lucienne. Falls back to `amd-smi` when
PCI IDs are ambiguous.

Override if needed:
```bash
LLAMA_GFX_VERSION_OVERRIDE=11.0.3 ./llama-run.sh --server
```

The script also detects your CPU ISA level and generates optimal
cmake flags so ISA extensions - AVX-512 BF16 on Zen 4, AVX2 on
Zen 3 - are enabled at build time.

## Features

### Auto-profiling

Models are detected by filename and assigned a profile automatically -
no manual configuration needed. MoE models get checkpoint strategies
and reasoning format support. SSM/Mamba models get context-shift
disabled. Large dense models get optimized batch sizes. The profile
is logged at startup (e.g. `Auto profile: moe-optimized (20GB,
MoE=true, SSM=false)`).

### Hybrid MoE architectures

Qwen3.6 (both dense and MoE variants), Qwen3-Coder-Next, and
Qwen3.5-122B mix attention layers with recurrent (Mamba) layers.
Laguna-S-2.1 is a DFlash target with shared experts and sliding
window. CachyLLama handles these correctly: KV cache shifting,
attention-only memory clearing that preserves recurrent state, and
checkpoint overflow prevention - so same-conversation state is always
accepted regardless of size.

### MoE expert tracking

MoE models activate only a subset of experts per token - typically
3-8 out of 128-256. CachyLLama tracks which experts activate through
`GET /expert-stats` and `POST /expert-tracking` endpoints. This is
instrumentation for now; future work will use it to reorder experts
for cache locality.

### MoE expert residency (run models larger than RAM)

Standard loading OOMs when a MoE model exceeds physical memory, even
though only 1-3% of its weights are touched per token. CachyLLama's
residency subsystem keeps only the active subset paged in via
`madvise(MADV_WILLNEED / MADV_DONTNEED)` and lets cold pages spill
to SSD. On the Flip KB (25 GB RAM), this loads and runs MoE models up
to 86 GB (Qwen3-Coder-Next Q8_K_XL).

A per-layer recency+frequency cache scores and evicts cold experts
(FlashMoE showed pure LRU evicts hot experts 34% of the time).
A co-activation matrix persists across sessions at
`~/.cachylla/coactivation/{model}.json` and informs future prewarm.

Enabled in `llama-run.sh` automatically for the MoE profile when the
handheld tier is plugged into AC (heavy SSD I/O drains battery).
CLI: `--moe-expert-residency [--moe-resident-per-layer N] [--moe-prewarm-top-k N]`.
See [CachyLLama docs/moe-expert-residency.md](CachyLLama/docs/moe-expert-residency.md)
for architecture, API, tuning, and limitations.

### User isolation

Multi-tenant deployments get per-user KV cache namespacing,
slot affinity (prefer slots already owned by the same user), and
configurable concurrency caps that return HTTP 429 when exceeded.
Pass `llama_user_id` in the request body to enable.

### CLIO integration

CLIO serializes tool definitions with deterministic JSON key ordering
and reuses conversation state across agentic turns. System prompts,
tool descriptions, and compressed context - the static content sent on
every API call - are cached and persisted to disk so they're available
immediately on restart.

### SSD-backed KV cache

The feature that makes all of this practical on APU hardware. Cache
hits skip prompt evaluation entirely for the cached prefix, dropping
TTFT from minutes to seconds. See the [caching deep-dive](#how-caching-works)
below for how it works in practice.

## How caching works

Agentic AI's bottleneck is prompt evaluation. Every API call sends
system prompt, tool definitions, and conversation history. Without
caching, all of it gets re-evaluated from scratch on every turn. An
18-30K-token prompt means minutes of waiting before the model can
start generating.

The SSD-backed KV cache persists conversation state across server
restarts. Only genuinely new content since the last turn needs fresh
evaluation. The rest loads from disk in 1-5 seconds.

### Cache tiers

- **Hot** - Current session, in RAM. Instant restore within the same
  conversation.
- **Warm** - Previous sessions, same server run. In RAM until memory
  pressure pushes them to cold.
- **Cold** - On disk. Survives server restarts.

Tiers promote and demote automatically based on turn activity. Old
checkpoints are evicted when the ring buffer fills. The cache just
works - no manual management needed.

By default nothing caps total disk usage - a long MoE conversation
can grow to hundreds of GiB on its own. Set `--cache-ssd-cold-maxsize`
(MiB) to bound the global cold tier footprint; when the cap is hit,
the oldest conversation directories get evicted first. Pair it with
`--cache-ssd-max-cold` for per-conversation checkpoint count limits.

### Per-tier policy

The cache is configured per hardware tier. On memory-constrained
hardware (handheld Flip KB, 26 GB OS-visible) the SSD tiers pay for
themselves - cold TTFT drops from minutes to seconds. On Strix Halo
(96 GB VRAM) the SSD cache is **disabled** for every profile (SSM,
MoE, large-dense, medium-dense, small). The 96 GB VRAM budget holds
the working set comfortably, and SSD writes are pure overhead on the
prefill critical path (3 SSD writes per turn at 62 MiB each = 4s of
disk time on a 15K-token prefill). In-memory checkpoints (`--cache-ram`
plus `--ctx-checkpoints`) handle prefix reuse within a server run; cold
TTFT is already fast enough that disk checkpoints don't pay off.

The `--no-ssd-cache` flag forces the same behavior on any tier if you
want to skip SSD caching manually.

### System prompt cache

A global cross-conversation cache stores the system section of any
prompt after first evaluation. On a warm server restart, the cache
restores the entire system prompt in under a second - the difference
between instant response and waiting minutes for the first turn.

The cache lives at `kv-cache/{model-stem}/sys-{hash}.bin`, keyed by
the first N tokens of the prompt. Defaults to 8 entries per model
with 30-day expiry.

## Benchmarks

### Strix Halo (Nimo Axis N161)

Radeon 8060S, 96GB APU VRAM, Vulkan backend. Large prompt results
only - [full data](benchmarks/20260810-0548/) with all sizes and
per-model breakdowns. Speedup is prompt eval speedup: cold `prompt_ms`
divided by warm `prompt_ms` (with warm path restoring the prefix from
SSD rather than re-evaluating it).

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

**MoE vs dense.** Qwen3.6-27B is the only dense model in the table -
all 27B parameters activate per token, trading generation speed
(4.7-4.8 t/s) for prompt eval throughput. MoE models activate
2.6-22B parameters per token, hitting 9-32 t/s. Dense models show the
largest speedup numbers from SSD caching (238x) because cold eval is
proportionally slower, but generation speed is the real-world tradeoff.

**Mamba hybrids.** Qwen3.6 (both dense and MoE variants),
Qwen3-Coder-Next, and Qwen3.5-122B interleave attention layers with
recurrent (Mamba-2) layers. These require attention-only KV cache
clearing to preserve recurrent state across cache restores. Qwen3-235B
(qwen3moe) is pure attention MoE - no Mamba. GLM-4.7 uses MLA (multi-head
latent attention, no Mamba), Gemma 4 uses interleaved
sliding-window/global attention, and DeepSeek-V4-Flash uses Lightning
Indexer. None of these are Mamba hybrids.

**Quantization.** Most models run at Q8_K_XL (30-82GB on disk).
Qwen3.5-122B-A10B at Q4_K_XL hits 32.7 t/s (vs 9.5 t/s at Q5_K_M) due
to lower memory-bandwidth pressure on decode. MiniMax-M2.7 at Q2_K_XL
pushes the frontier - 230B parameters, 256 experts, 4.9B active per
token, running in 96GB VRAM. DeepSeek-V4-Flash IQ3_XXS sits at the top
of the model size chart at ~1.6T parameters (256 experts, 8.4B active).

All warm runs restore the cached prefix from SSD; cold from scratch.
Warm TTFT stays under 1.5s across most models - MiniMax-M2.7 at 3.19s
is the outlier (its 300 MiB warm-path SSD payload for a 15K-token q8_0
KV cache costs more on the SSD restore path than smaller models).
See [benchmarks/20260810-0548/](benchmarks/20260810-0548/) for full
results with per-model breakdowns and all prompt sizes.

### Ayaneo Flip KB

Radeon 780M, 32GB unified memory, Vulkan backend. Large prompt
results - [full data](benchmarks/20260807-1921/) and
[benchmarks/20260721-1737/](benchmarks/20260721-1737/) with all
sizes.

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| gemma-4-26B Q5_K_M (26B, dense) | 63.8s | 0.46s | **138.4x** | 14-17 |
| gpt-oss-20b Q6_K_XL (20B, 3.6B active MoE) | 45.9s | 0.24s | **193.5x** | 25-31 |
| Qwen3.6-35B-A3B Q4_K_XL (35B, 3B active MoE) | 70.9s | 0.38s | **187.9x** | 19-24 |

Gemma 4 generation nearly tripled compared to the previous Flip
benchmark (6 -> 14-17 t/s), with cold TTFT also improving 33%
(94.8s -> 63.8s). Qwen3.6-35B-A3B cold eval improved 16% (84s ->
70.9s) while generation maintained and extended its range (19-24 t/s).

MoE expert residency keeps hot experts paged in via
`madvise(MADV_WILLNEED)`, so cold-path expert loads no longer
dominate decode. All warm TTFTs converge under 500ms - the SSD
cache bridges the hardware gap between Flip KB and Strix Halo on
subsequent turns.

### Real-world CLIO performance (Strix Halo)

CLIO sends 18-30K tokens of system prompt, tool definitions, and
conversation context on every API call. Every prompt has a static
prefix (~18K tokens - system prompt, tool definitions) that's
identical across all turns, and a dynamic tail that grows as the
conversation progresses.

This test has CLIO evaluate a project: read files, check git history,
run commands, write a final analysis. Qwen3.6-35B Q8_K_XL, 196K
context, 32 threads, Vulkan backend.

| Turn | Prompt tokens | TTFT | Gen t/s |
|------|---------------|------|---------|
| T0 | 21,336 | 26.5s | 36.5 |
| T1 | 1,623 | 2.8s | 37.9 |
| T2 | 5,152 | 8.0s | 37.7 |
| T3 | 4,428 | 7.5s | 38.2 |

**Total: 88 seconds** (32,539 prompt tokens, 1,656 generated). First
turn evaluates the full system prompt cold. Every subsequent turn uses
in-memory checkpoints to restore the cached prefix, evaluating only
the new content. Generation holds steady at 36-38 t/s across all turns.

## Repo structure

```
├── llama-run.sh              # Main entry point
├── CachyLLama/               # Submodule - our fork of llama.cpp
├── scripts/
│   ├── rebuild.sh            # Build script (Vulkan default, optional ROCm)
│   ├── env.sh                # Environment setup
│   ├── detect-gpu.sh         # GPU/APU and CPU ISA auto-detection
│   ├── benchmark.sh          # KV cache performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config
├── src/
│   ├── cachy-llama-vulkan/   # Vulkan build output + build.sh
│   ├── cachy-llama-rocm/     # ROCm build output + build.sh
│   └── cachy-llama-metal/    # Metal build output + build.sh (macOS)
├── deps/                     # ROCm SDK (downloaded by rebuild.sh)
├── models/                   # GGUF files
├── kv-cache/                 # SSD-backed KV cache
├── scratch/                  # Transient working files
└── benchmarks/               # Results with full server logs
```

See [AGENTS.md](AGENTS.md) for the technical reference (build commands,
code style, directory conventions).

## License

Source code: [GPL-3.0-or-later](LICENSE)
Documentation: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

llama.cpp is MIT-licensed. ROCm components carry AMD's license.
