Local LLM inference on AMD APU hardware. Built around
[CachyLLama](https://github.com/fewtarius/CachyLLama), our fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp), with an SSD-backed
KV cache, agentic workflow tuning, and tight
[CLIO](https://github.com/SyntheticAutonomicMind/CLIO) integration.
Self-contained - no system ROCm install required. Vulkan (RADV) is the
default backend for best stability on RDNA3 iGPUs.

## Introduction

llama-ai is a deployment of CachyLLama aimed at running hybrid MoE
models on AMD APUs. The primary target is the Strix Halo "max"
platform (Ryzen AI Max+ 395, Radeon 8060S, 128GB unified memory with
96GB pre-allocated to the APU). It also runs on the Ayaneo Flip KB
(7840U / Radeon 780M / 32GB) and the Minisforum UM580 (5800H / 16GB),
and any other AMD APU in the supported detection map.

Profiles scale automatically with the APU's VRAM carveout via
`LLAMA_HARDWARE_TIER` (`handheld` / `standard` / `halo`): halo gets
192K-token context on MoE models with fp16 KV cache and 16GB cache-ram,
handheld keeps the conservative 64K-token / q8_0 / 6GB settings tuned
for the 780M's 6GB VRAM envelope.

The fork exists so performance work on hybrid architectures (Qwen3.5/3.6,
GLM-4.7, Gemma 4) lives as code in the
[CachyLLama](https://github.com/fewtarius/CachyLLama) git history rather
than as patches layered on releases. The submodule in this repo points
at the fork, not at ggml-org/llama.cpp.

On top of CachyLLama, llama-ai adds:

- SSD-backed KV cache that survives reboots and power outages, with
  hot/warm/cold tiering and a global system prompt cache
- CLIO integration tuned for cache reuse across agentic turns
  (deterministic JSON serialization, slot affinity, per-user isolation)
- Auto GPU and CPU ISA detection for AMD APUs across generations
- A benchmarking harness for measuring prompt-eval speedup
- Auto-profile model selection and a runner that strips reasoning
  blocks to keep prompt tokens small

## Why

The goal is reasonably-performing agentic AI development on AMD APU
hardware - usable when there is no network. No API keys, no per-token
costs, no cloud dependency. Primary target is the Strix Halo "max"
platform (Ryzen AI Max+ 395 / Radeon 8060S / 96GB APU VRAM / 128GB
total). Also runs well on the [Ayaneo Flip KB](https://ayaneo.com/product/AYANEO-FLIP-KB)
(7840U / 32GB) and similar Zen 4 APUs.

<p align="center">
  <img src=".images/flip.webp" alt="Ayaneo Flip KB" width="800">
</p>

[CLIO](https://github.com/SyntheticAutonomicMind/CLIO) is optimized for this implementation. It serializes tool definitions with deterministic JSON key ordering and reuses conversation state to maximize cache hits across agentic turns. System prompts, tool descriptions, and compressed context - the static content sent on every API call - are cached and persisted to disk so they're available immediately on the next request.

## Table of contents

- [Introduction](#introduction)
- [Why](#why)
- [Quick start](#quick-start)
- [Backends](#backends)
  - [Vulkan (Linux/AMD)](#vulkan-linuxamd)
  - [ROCm (Linux/AMD)](#rocm-linuxamd)
  - [Metal (macOS)](#metal-macos)
- [GPU memory](#gpu-memory)
- [GPU detection](#gpu-detection)
  - [CPU ISA detection](#cpu-isa-detection)
- [Usage](#usage)
- [How it works](#how-it-works)
  - [Auto-profiling](#auto-profiling)
  - [KV cache](#kv-cache)
    - [Hot/warm/cold tiering](#hotwarmcold-tiering)
    - [Search strategy](#search-strategy)
    - [System prompt cache](#system-prompt-cache)
    - [Kernel readahead](#kernel-readahead)
  - [User isolation](#user-isolation)
  - [MoE expert tracking](#moe-expert-tracking)
- [Benchmarking](#benchmarking)
  - [Test methodology](#test-methodology)
  - [Results](#results)
  - [Running the benchmark](#running-the-benchmark)
  - [Output](#output)
- [Real-world CLIO performance](#real-world-clio-performance)
  - [Workload profile](#workload-profile)
  - [Test scenario](#test-scenario)
  - [Results - cold start](#results-cold-start-no-cache)
  - [Results - warm restart](#results-warm-restart-cache-populated)
  - [Takeaways](#takeaways)
- [What CachyLLama adds](#what-cachyllama-adds)
- [Structure](#structure)
- [License](#license)

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

## Backends

### Vulkan (Linux/AMD)

Default backend. Uses the Mesa RADV driver - no ROCm install required. Best stability on RDNA3 iGPUs (Phoenix, Hawk Point, Strix Point) and earlier GCN/RDNA generations. CPU offloading works for models that don't fit in GPU memory.

### ROCm (Linux/AMD)

Optional. Has known stability issues on some architectures - GLM-4.7-Flash and DeepSeek2 MLA models produce zero generation tokens on RDNA3. Use Vulkan unless you have a specific reason to try ROCm.

### Metal (macOS)

Apple Silicon (M1/M2/M3/M4) and Intel Macs with Metal-capable GPUs. Build with `./scripts/rebuild.sh` on macOS - it auto-detects the platform and builds the Metal backend.

## GPU memory

AMD APUs share system RAM with the GPU. Use `apply-ttm-kernel-params.sh`
to configure GTT:

```bash
# Phoenix/Hawk Point: cap firmware VRAM at 6GB, add 18GB GTT
sudo ./scripts/apply-ttm-kernel-params.sh 18

# Strix Halo: BIOS already pre-allocates the VRAM carveout, just leave
# the script at its auto-detected defaults (vis_vramlimit is skipped
# entirely so the BIOS allocation is preserved)
sudo ./scripts/apply-ttm-kernel-params.sh
sudo reboot
```

Writes kernel parameters (`amdgpu.gttsize`, `amdgpu.vis_vramlimit`,
`ttm.pages_limit`) to your bootloader config. Also calls `amd-smi set -G`
as a runtime hint, but kernel parameters are the authoritative method
that persists across reboots.

Supports GRUB (SteamFork 3.7) and systemd-boot (SteamFork 3.8+). Tested
on JELOS - should work with any distro that exposes the AMD GPU through
sysfs and supports `amdgpu.vis_vramlimit` / `amdgpu.gttsize` kernel
parameters (i.e. most modern Linux distributions with a 6.x kernel).

GTT size and `vis_vramlimit` default to tier-aware values:

| Tier    | Examples                  | vis_vramlimit | GTT      |
|---------|---------------------------|---------------|----------|
| handheld| 780M, 890M (Phoenix/Hawk) | 6GB           | RAM-6GB  |
| standard| 16-32GB APU VRAM          | 16GB          | 8GB      |
| halo    | 8060S (Strix Halo)        | not set       | 4GB      |

The `halo` tier skips `vis_vramlimit` because the BIOS carveout
(typically 96GB) must be preserved - capping it would shrink the
addressable VRAM. Override with `VIS_VRAM_LIMIT_MB` env var or the
first positional argument.

Verify after reboot:
```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E "amdgpu|ttm"
```

## GPU detection

Auto-detects AMD GPU via PCI device ID and sets `HSA_OVERRIDE_GFX_VERSION`
for ROCm.

Supported: Cezanne (5800H), Phoenix (780M), Hawk Point (890M/780M),
Strix Point (890M/880M), Strix Halo (8060S), Sephiroth, Rembrandt
(680M/660M), Mendocino (610M), Renoir, Lucienne. Falls back to
`amd-smi` for authoritative detection when PCI IDs are ambiguous
(e.g. Cezanne and Van Gogh share the same PCI ID). To add your device,
edit the `GPU_MAP` in `scripts/detect-gpu.sh`.

Override detection:
```bash
LLAMA_GFX_VERSION_OVERRIDE=11.0.3 ./llama-run.sh --server
```

### CPU ISA detection

`detect-gpu.sh` also detects the CPU ISA level and generates optimal cmake flags:

| CPU | ISA Level | CMake Flags |
|-----|-----------|-------------|
| Zen 4 (7840U) | avx512_bf16 | `-DGGML_AVX512=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VNNI=ON` |
| Zen 3 (5800H) | avx2 | `-DGGML_AVX2=ON -DGGML_AVX=ON -DGGML_FMA=ON` |
| Apple Silicon | apple_silicon | (none - ARM NEON auto-detected) |

Previously, the Vulkan build was compiled with `GGML_NATIVE=OFF` and `GGML_AVX512=OFF`, leaving AVX-512 code paths compiled out on Zen 4 hardware that supports them. This cost 5-15% generation speed on Vulkan and 30-100% on CPU-offloaded layers. Now `rebuild.sh` uses `$LLAMA_CMAKE_CPU_FLAGS` to enable the right ISA level.

Override:
```bash
LLAMA_CPU_ISA_OVERRIDE=avx2 ./scripts/rebuild.sh
```

## Usage

```bash
# List models found in models/
./llama-run.sh --list-models

# Start server (auto-detects model, Vulkan backend)
./llama-run.sh --server

# Specific model and backend
./llama-run.sh --server gemma-4-26b --backend vulkan

# Download a model
./llama-run.sh --download Qwen3-14B --quant Q4_K_M

# List available backends
./llama-run.sh --list-backends

# Rebuild options
./scripts/rebuild.sh              # Vulkan only (default)
./scripts/rebuild.sh --rocm       # ROCm only
./scripts/rebuild.sh --both       # Vulkan + ROCm
./scripts/rebuild.sh --rebuild    # Full rebuild from scratch
```

Reasoning models (DeepSeek-R1, Qwen3.6, GLM-4.7) emit thinking blocks before each response. By default the runner strips these from prior assistant messages in the conversation history so they don't waste prompt tokens. To preserve them across turns (some workflows benefit from this), pass `--preserve-reasoning`. The `--reasoning-budget N` flag caps thinking tokens per response (default: 2048) to prevent runaway generation.

## How it works

### Auto-profiling

Models are auto-profiled based on filename characteristics. MoE models get checkpoint strategies and reasoning format; SSM/Mamba models get context-shift disabled; large dense models get optimized batch sizes. Profiles are assigned dynamically - no hard-coded model names. The profile name is logged at server startup (e.g. `Auto profile: moe-optimized (20GB, MoE=true, SSM=false)`).

### KV cache

SSD-backed KV cache persists conversation state across server restarts. Enabled by default for all non-SSM models. The cache directory is `kv-cache/`. When a `user_id` is supplied (see [User isolation](#user-isolation)), checkpoints route to a separate `kv-cache/u/` namespace.

#### Hot/warm/cold tiering

The cache has three tiers with automatic promotion and demotion:

- **Hot tier** - Checkpoints from the current session, kept in RAM. Instant restore when the same conversation continues. After 2 turns of inactivity, hot checkpoints are demoted to warm.
- **Warm tier** - Checkpoints from previous sessions in the same server run. In RAM until memory pressure forces demotion to cold. After 4 turns of inactivity, warm checkpoints are demoted to cold.
- **Cold tier** - On-disk checkpoints with token prefixes. Survives server restarts. Each conversation gets up to the ring buffer limit of cold checkpoints on disk. When the limit is exceeded, the oldest cold checkpoint is deleted. Up to 16 conversations are tracked simultaneously (configurable with `--cache-ssd-max-conversations`).

#### System prompt cache

The system prompt cache is a global (cross-conversation) cache that stores the system section of any prompt after first evaluation. On cold start - server restart, first request, or a model that has not been seen before - the server checks the system prompt cache before falling through to full evaluation. A hit returns the cached state directly, skipping the entire system prompt re-eval.

The cache lives at `{kv-cache-path}/{model-stem}/sys-{hash}.bin`. Entries are keyed by the first N tokens of the prompt (the system section) and stored with a model compatibility hash that rejects mismatches on load. Default: 8 entries per model, 30 days unused before expiry. Override with `--cache-ssd-system-prompts N` and `--cache-ssd-system-max-days N`.

The system prompt cache works for both standard transformer and hybrid (MoE/SSM) models. For hybrid architectures, the recurrent state is stored per-position in the state file, so a state saved after processing the full prompt can be restored with `n_past` capped to the system prompt boundary - the inference engine reads the cell at that position regardless of how many tokens came after.

#### Search strategy

When an API request arrives, the server searches for a matching checkpoint in three stages:

1. **Same-conversation** (Tier 1) - Matches by conversation hash (`conv_hash`), a FNV-1a hash of the first 1024 task tokens. This finds the checkpoint from a previous turn of the same conversation. Fast, accurate, and the most common hit path.

2. **Shared prefix** (Tier 2) - Cross-conversation match using `n_past` (the common prefix length). This reuses cached system prompt evaluation across different conversations with the same model. Works because the first N tokens are identical - tool definitions, system instructions, etc.

3. **Cold-start token prefix** (Tier 3) - Used on server restart when `n_past == 0`. The server compares the prompt's first tokens against every checkpoint's stored token prefix (up to 4096 tokens per checkpoint). This has two phases:
   - **Chain match** - Same conversation, full prefix matches. The largest checkpoint from the same conversation is preferred, even if it's large - the recurrent state is content-accurate.
   - **Safe match** - Cross-conversation or partial prefix. Only checkpoints whose `n_tokens` fits within the common prefix (LCP) are considered. This avoids restoring recurrent state computed from different conversation content.

Overflow handling differs by match type. Same-conversation checkpoints (Tier 1 and Tier 3 chain) skip size and staleness checks entirely - the recurrent state is content-accurate, so any same-conv checkpoint is valid. If the checkpoint covers more tokens than the current task, `n_past` is capped in the restore layer to leave room for new token evaluation instead of resetting. Cross-conversation matches (Tier 2 and Tier 3 safe) skip oversized checkpoints at the search layer, since the recurrent state was computed from different conversation content.

Each checkpoint is stored as a separate file (`ckpt-N.bin`) in `kv-cache/{conv_hash}/` with metadata in `index.bin`. Turn tracking survives server restarts - the next turn counter is seeded from the maximum turn ID found on disk, so warm-tier entries from a previous server run start aging from turn 0 of the new run rather than being immediately demoted.

Every checkpoint carries:
- `conv_hash` - Conversation identity (first 1024 tokens)
- `compat_hash` - Model configuration hash (architecture, dimensions, cache types). Checkpoints with mismatched compat hashes are rejected, preventing silent corruption when switching between models.
- `token_prefix` - First 4096 tokens for cold-start prefix matching
- `turn_id` - Tracks when the checkpoint was last accessed for tier management

#### Kernel readahead

When a cold checkpoint is identified for loading, the server issues `posix_fadvise(POSIX_FADV_WILLNEED)` on Linux (or `readahead()` on macOS) to trigger kernel page cache prefetch. This overlaps SSD I/O with CPU work (token matching, state restoration setup) and reduces cold TTFT by ~0.5-0.75s for typical checkpoint sizes.

#### What happens on cache hit

The KV cache (attention state) and recurrent state (for hybrid MoE models) are restored from the checkpoint. Only tokens beyond the checkpoint's coverage need evaluation. A 18-30K-token prompt might need just a handful of new tokens evaluated - the rest is restored from disk in 1-5 seconds depending on checkpoint size.

The cache is persisted automatically after each turn. No manual management needed.

### User isolation

Multi-tenant deployments need isolation between users sharing the same server. This fork adds three dimensions of isolation:

#### Identity

The `user_id` field is a first-class request parameter. Pass it in the request body:

```json
{
  "model": "...",
  "messages": [...],
  "llama_user_id": "tenant-42-user-7"
}
```

OpenAI SDK callers pass it through `extra_body`:

```python
client.chat.completions.create(
    model="...",
    messages=[...],
    extra_body={"llama_user_id": "tenant-42-user-7"},
)
```

Validated to `^[a-zA-Z0-9\-_]+$` with a 512-char ceiling. Empty string is valid (anonymous bucket).

#### KV cache routing

When `user_id` is present, the SSD page manager routes checkpoints to a separate `u/` namespace on disk:

```
{ssd_path}/{hash_hex}/    # anonymous (conv_hash)
{ssd_path}/u/{hash_hex}/  # user-scoped (fnv1a(user_id))
```

Cross-user lookup is disabled for user-scoped requests. A user can only access their own cached state, never another user's directory.

#### Scheduling isolation

`--max-concurrent-per-user N` caps the number of simultaneous slots a single user_id can occupy. When the cap is hit, the server returns HTTP 429 with a `rate_limit_error` type:

```json
{
  "error": {
    "code": 429,
    "message": "per-user concurrency cap reached for user_id=tenant-42-user-7",
    "type": "rate_limit_error"
  }
}
```

Slot allocation also prefers slots already owned by the requesting user (cache affinity). An empty slot (post-release) is fair game for any user.

Default: 0 (unlimited). Set to 1 for strict one-at-a-time, or 2-3 for concurrent with backpressure.

Design rationale: [`docs/development/user-isolation-design.md`](CachyLLama/docs/development/user-isolation-design.md)

### MoE expert tracking

MoE models (Qwen3.5/3.6, Gemma 4, GLM-4.7) activate only a subset of experts per token. This fork adds real-time expert activation tracking via two HTTP endpoints:

#### GET /expert-stats

Returns per-layer expert activation counts, frequencies, and token counts:

```json
{
  "n_expert": 256,
  "n_expert_used": 8,
  "total_tokens": 1500,
  "tracking_enabled": true,
  "layers": [
    {
      "layer": 0,
      "activations": [
        {"expert": 42, "count": 150, "frequency": 0.0125},
        {"expert": 7, "count": 148, "frequency": 0.0123},
        ...
      ]
    },
    ...
  ]
}
```

#### POST /expert-tracking

Enable/disable tracking and optionally reset counters:

```json
{"enabled": true, "reset": true}
```

This is Phase 1 of the MoE expert tiering design - instrumentation only, no compute changes. Future phases will use this data to reorder experts for cache locality and offload cold experts to RAM/SSD.

## Benchmarking

The bottleneck in agentic AI isn't generation speed (the model produces tokens as fast as the GPU allows). The bottleneck is **prompt evaluation** - reprocessing the entire prompt before the model can generate its first token.

Every API call in an agentic workflow sends static content: system prompt, tool definitions, prior conversation context. Without caching, this content is re-evaluated from scratch on every single call. An 18-30K-token prompt means it could be several minutes before the model starts responding on an APU like the 780M. With SSD cache and a 17,800-token prefix hit, only the divergent tail of the prompt is evaluated - typically a few seconds when only the latest tool result is new.

### Test methodology

Real agentic workloads send 12-20K tokens of system prompt and tool definitions on every API call, growing to 32-64K tokens with compressed conversation context. Every token is re-evaluated from scratch without caching.

The benchmark uses scaled-down prompts to demonstrate cache mechanics and prove the speedup is real. The same principles apply at production sizes - speedup ratios increase with prompt length.

| Size | Tokens | What it measures |
|------|--------|-----------------|
| Small | ~1,100 | Cache overhead and baseline speedup |
| Medium | ~5,200 | Checkpoint matching and partial restore |
| Large | ~15,500 | Full checkpoint restore with large prefix |

Each size runs twice:

1. **Cold** - Empty cache, server starts fresh. The entire prompt is evaluated from scratch.
2. **Warm** - Server restarts with existing SSD cache. The server restores the matching checkpoint from disk and evaluates only the delta.

The key metric is **TTFT** (Time To First Token) - how long before the model starts generating. Generation speed doesn't change with caching (same model, same hardware). What changes is the wait before generation begins.

### Results

Benchmarks run on the Strix Halo "max" platform (Ryzen AI Max+ 395,
Radeon 8060S, 128GB unified memory, 96GB APU VRAM) and the Ayaneo Flip
KB (7840U / 780M / 32GB / Vulkan). Both use the same Vulkan backend
(Mesa RADV) and the same SSD cache machinery - the only thing that
changes is the underlying compute, memory, and context size.

#### Strix Halo

Radeon 8060S, 96GB APU VRAM, ctx 196608 (MoE) / 32768 (dense), 128
output tokens, all GPU layers. Three benchmark runs: an initial run
with aggressive SSD checkpointing (3 checkpoints per prefill, 24
in-memory ring, 64 on-disk), a tuned run with minimal SSD
checkpointing (1-2 per prefill, 8 in-memory ring, 8 on-disk), and a
final run after fixing the system-prompt boundary detector for
GLM/Gemma templates. Full per-test data:
[`benchmarks/20260620-1314/`](benchmarks/20260620-1314/) (aggressive),
[`benchmarks/20260620-1433/`](benchmarks/20260620-1433/) (tuned), and
[`benchmarks/20260620-1639/`](benchmarks/20260620-1639/) (boundary fix).

| Model | Size | Cold TTFT | Warm TTFT | Speedup | Cached |
|-------|------|-----------|-----------|---------|--------|
| Qwen3.6-35B-A3B Q4_K_XL (boundary fix) | small (~1.2K) | 2.0s | 0.21s | **1.7x** | 1237/1243 |
| Qwen3.6-35B-A3B Q4_K_XL (boundary fix) | medium (~5.4K) | 7.7s | 0.42s | **3.5x** | 5403/5409 |
| Qwen3.6-35B-A3B Q4_K_XL (boundary fix) | large (~15.7K) | 23.6s | 0.75s | **7.8x** | 15715/15721 |
| Qwen3.6-35B-A3B Q4_K_XL (tuned) | small (~1.2K) | 1.8s | 0.22s | **1.6x** | 1237/1243 |
| Qwen3.6-35B-A3B Q4_K_XL (tuned) | medium (~5.4K) | 7.0s | 0.30s | **3.4x** | 5403/5409 |
| Qwen3.6-35B-A3B Q4_K_XL (tuned) | large (~15.7K) | 21.4s | 0.49s | **7.85x** | 15715/15721 |
| Qwen3.6-35B-A3B Q4_K_XL (aggressive) | small (~1.2K) | 1.8s | 0.25s | **1.63x** | 1237/1243 |
| Qwen3.6-35B-A3B Q4_K_XL (aggressive) | medium (~5.4K) | 7.1s | 0.30s | **3.52x** | 5403/5409 |
| Qwen3.6-35B-A3B Q4_K_XL (aggressive) | large (~15.7K) | 22.2s | 2.3s | **5.13x** | 15715/15721 |
| GLM-4.7-Flash Q4_K_M (boundary fix, ctx 192K) | small (~1.1K) | 2.1s | 0.15s | **1.8x** | 1237/1243 |
| GLM-4.7-Flash Q4_K_M (boundary fix, ctx 192K) | medium (~5.2K) | 11.8s | 0.53s | **4.7x** | 5403/5409 |
| GLM-4.7-Flash Q4_K_M (boundary fix, ctx 192K) | large (~15.5K) | 59.6s | 1.59s | **12.6x** | 15715/15721 |
| GLM-4.7-Flash Q4_K_M (ctx 32K) | small (~1.1K) | 2.4s | 2.0s | 1.12x | 3/1145 |
| GLM-4.7-Flash Q4_K_M (ctx 32K) | medium (~5.2K) | 11.0s | 10.8s | 1.01x | 3/5237 |
| GLM-4.7-Flash Q4_K_M (ctx 32K) | large (~15.5K) | 55.2s | 54.7s | 1.01x | 3/15489 |
| gemma-4-26B-A4B Q5_K_M (boundary fix, ctx 192K) | small (~1.4K) | 2.2s | 0.30s | **1.6x** | 1237/1413 |
| gemma-4-26B-A4B Q5_K_M (boundary fix, ctx 192K) | medium (~6.1K) | 9.3s | 0.32s | **3.6x** | 6080/6083 |
| gemma-4-26B-A4B Q5_K_M (boundary fix, ctx 192K) | large (~17.3K) | 31.0s | 0.54s | **8.7x** | 17343/17347 |
| gemma-4-26B-A4B Q5_K_M (ctx 32K) | small (~1.4K) | 5.8s | 2.0s | 1.78x | 7/1413 |
| gemma-4-26B-A4B Q5_K_M (ctx 32K) | medium (~6.1K) | 8.9s | 8.6s | 1.03x | 7/6083 |
| gemma-4-26B-A4B Q5_K_M (ctx 32K) | large (~17.3K) | 29.7s | 28.8s | 1.03x | 7/17347 |

The tuned profile (halo tier) for Qwen3.6-35B-A3B pushes warm TTFT
from 2.3s down to 0.49s on large prompts - a **4.7x improvement on the
warm path** - by writing fewer per-turn checkpoints to SSD and letting
the in-memory ring stay small. The boundary-fix run for Qwen3.6 is
slightly slower on the warm path (0.75s vs 0.49s) because the
per-conversation SSD checkpoint now does the restore instead of the
system prompt cache incorrectly storing the entire prompt. This is
the correct mechanism - the SSD cache handles same-conversation
restarts, the system prompt cache handles cross-conversation reuse.
See [Boundary detection fix](#boundary-detection-fix) below.

Cold prompt eval: 280-735 t/s on the Strix Halo (vs 33-166 t/s on the
Ayaneo Flip KB). The hybrid MoE architectures (Qwen3.6, GLM-4.7-Flash)
restore both attention KV state and recurrent state from disk - the
Mamba layers are checkpoint-aware and the cache works across restarts.

#### Boundary detection fix

The system prompt cache (cross-conversation, keyed by the first N
tokens of the system section) and the per-conversation SSD cache
(same conversation, keyed by conv_hash) are two separate mechanisms
that both help warm-restart speed. The boundary detector
(`kv_detect_system_prompt_boundary`) had a bug that affected GLM-4
and Gemma: both vocabularies classify role tokens
(`<|user|>`, `<|assistant|>`, `<|system|>`) as EOG. The original
detector scanned for the first EOG after the role header, found the
first role-marker token, and returned n_sys=2-7 (just the chat
template header). This made the system prompt cache useless for
GLM/gemma (warm TTFT was effectively unchanged from cold).

The fix is a two-phase detector:
1. **Find first user-role marker** by decoded text - works for
   templates without explicit section close markers (GLM, Gemma,
   Command-R). Filters content-word false positives by requiring
   the preceding token to be a control token (e.g. `<|im_start|>user`
   in ChatML, not `user` as a word in a novel).
2. **EOG-based detection** with role-marker EOGs filtered out,
   capped at 64 tokens to prevent finding the EOG that closes the
   USER section when no system section exists.

Returns `min(phase1, phase2)` for backwards compat with templates
that already worked (ChatML, Llama-3, standard Gemma).

A second guard in `try_restore_system_prompt` skips the system
prompt cache when the detected boundary is < 16 tokens (just the
chat template header, not a meaningful system prompt). This lets
the per-conversation SSD cache handle the restore, which is the
correct mechanism for same-conversation warm restarts.

Result: GLM large warm TTFT went from 54.7s (1.01x speedup) to
1.59s (12.6x speedup), gemma large from 28.8s (1.03x) to 0.54s
(8.7x). Qwen3.6 large is essentially unchanged (7.8x vs 7.85x)
because it was already benefiting from the SSD cache - the buggy
system prompt cache was just masking that mechanism.

#### Halo SSD caching strategy

The Strix Halo's 96GB APU VRAM carveout changes the SSD caching
tradeoff versus memory-constrained APUs (Ayaneo Flip KB: 6GB VRAM +
18GB GTT). On Halo, a 35B MoE with 192K context at f16 KV uses ~50GB
of VRAM - it fits comfortably in the 96GB ceiling with headroom for
the in-memory prompt cache and the working set. The SSD cache exists
on Halo for two reasons and two reasons only:

1. **System prompt cache** (cross-restart) - the global cache at
   `{ssd-path}/{model-stem}/sys-{hash}.bin`. One entry per distinct
   system prompt, restores 15-18K tokens of system + tool definitions
   in 0.5s. This is the warm-path win.
2. **Long single-conversation runs** - if a single chat thread grows
   past what VRAM can hold, SSD checkpoints provide eviction
   insurance. In practice this rarely happens on Halo.

The aggressive checkpoint strategy from the Flip KB tier (24-entry
in-memory ring, 64-entry on-disk ring, 8K checkpoint interval) was
inherited from the Flip KB profile. On Halo it produced 3 SSD writes
per 15K-token prefill (~200 MiB on the disk critical path) for
protection VRAM already provides. The tuned profile:

- `--checkpoint-every-n-tokens 16384` (was 8192) - 1 checkpoint for
  a typical 8-15K system prompt, 0 for short prompts
- `--ctx-checkpoints 8` (was 24) - the in-memory ring stays small
- `--cache-ssd-checkpoints 8` (was 64) - cap the on-disk ring, no need
  to keep stale per-turn checkpoints around
- `--cache-ram 16384` (unchanged) - 16GB in-RAM prompt cache, the
  primary cache layer for warm within-server restarts

Result: large-prompt warm TTFT dropped from 2.3s to 0.49s (4.7x
improvement) by avoiding the restore of 2 stale per-turn checkpoints
from disk. Disk writes per prefill dropped from 3 to 1. Cold TTFT is
unchanged because cold has no checkpoints to restore.

The system prompt cache (the only cross-restart disk cache that
genuinely helps on Halo) is unchanged.

#### Ayaneo Flip KB

Radeon 780M, 6GB VRAM + 18GB GTT, ctx 32768, 128 output tokens, all
GPU layers. 2026-05-03 baseline. Full per-test data:
[`benchmarks/20260611-0656/`](benchmarks/20260611-0656/).

#### GLM-4.7-Flash (Q4_K_M, 30B MoE, 3B active)

| Size | Tokens | Cold TTFT | Warm TTFT | Speedup | Gen TPS |
|------|--------|-----------|-----------|---------|---------|
| Small | ~1,145 | 9.7s | 0.34s | 28.4x | 20.2 |
| Medium | ~5,237 | 74.2s (1.2min) | 1.0s | 72.7x | 12.1 |
| Large | ~15.5K | 467.6s (7.8min) | 2.7s | 174.1x | 5.7 |

Cold prompt eval: 33.1-117.6 t/s. Cached: 15,485/15,489 tokens at large size (4 tokens evaluated on warm).

#### Gemma 4 26B (Q5_K_M, 26B MoE, 4B active)

| Size | Tokens | Cold TTFT | Warm TTFT | Speedup | Gen TPS |
|------|--------|-----------|-----------|---------|---------|
| Small | ~1,413 | 8.5s | 0.71s | 12.0x | 16.2 |
| Medium | ~6,083 | 38.0s | 0.97s | 39.2x | 15.3 |
| Large | ~17.3K | 130.9s (2.2min) | 1.4s | 92.9x | 13.8 |

Cold prompt eval: 132.6-165.6 t/s. Cached: 17,343/17,347 tokens at large size (4 tokens evaluated on warm).

#### Qwen3.6-35B (Q4_K_XL, 35B MoE hybrid, 3B active)

| Size | Tokens | Cold TTFT | Warm TTFT | Speedup | Gen TPS |
|------|--------|-----------|-----------|---------|---------|
| Small | ~1,243 | 9.3s | 0.41s | 23.0x | 21.7 |
| Medium | ~5,409 | 43.3s | 0.57s | 76.2x | 20.5 |
| Large | ~15.7K | 143.1s (2.4min) | 0.99s | 144.5x | 18.6 |

Cold prompt eval: 109.9-133.4 t/s. Cached: 15,717/15,721 tokens at large size (4 tokens evaluated on warm).
35B parameters with only 3B active keeps the eval rate high. The SSD cache restores both attention KV state and recurrent state from disk - the hybrid architecture's Mamba layers are checkpoint-aware and restore correctly across restarts.

#### Summary

Strix Halo (top row per model) vs Ayaneo Flip KB (bottom row), large
prompt only. All Strix Halo numbers from the boundary-fix run
([`benchmarks/20260620-1639/`](benchmarks/20260620-1639/)) using the
tuned profile (see [Halo SSD caching strategy](#halo-ssd-caching-strategy)
and [Boundary detection fix](#boundary-detection-fix)).

| Model | Strix Halo cold | Strix Halo warm | Strix speedup | Flip cold | Flip warm | Flip speedup |
|-------|----------------:|----------------:|--------------:|----------:|----------:|-------------:|
| GLM-4.7-Flash | 59.6s | 1.59s | **12.6x** | 467.6s (7.8min) | 2.7s | 174.1x |
| Qwen3.6-35B | 23.6s | 0.75s | **7.8x** | 143.1s (2.4min) | 1.0s | 144.5x |
| gemma-4-26B | 31.0s | 0.54s | **8.7x** | 130.9s (2.2min) | 1.4s | 92.9x |

The Strix Halo's 8060S evaluates prompts 5-20x faster than the 780M, so
the absolute warm-cache wall time is much smaller. On the tuned Qwen3.6
profile the warm path is dominated by generation (488ms prompt eval +
2.5s gen for 128 tokens); the SSD read overhead is a tiny fraction of
wall time. In absolute terms the cache still saves ~21 seconds per
turn on Qwen3.6 long-context agentic workloads - and at 192K context
with fp16 KV, the cache layer is what makes that context window
actually usable in practice.

Full benchmark data (server logs, API responses, timing stats):
[`benchmarks/20260611-0656/`](benchmarks/20260611-0656/) (Ayaneo Flip KB)
and [`benchmarks/20260620-1314/`](benchmarks/20260620-1314/) (Strix Halo).

### Running the benchmark

```bash
# Full benchmark: all models, Vulkan backend
./scripts/benchmark.sh

# Single model
./scripts/benchmark.sh --model GLM-4.7-Flash-Q4_K_M.gguf

# Both backends
./scripts/benchmark.sh --backend both
```

Uses public domain text from The Count of Monte Cristo (Project Gutenberg), cached locally in `scratch/pg1184.txt`. Each prompt appends "Summarize this passage in one sentence." to keep generation short (128 tokens).

### Output

```
benchmarks/YYYYMMDD-HHMM/
├── vulkan/
│   ├── GLM-4.7-Flash-Q4_K_M/
│   │   ├── server-small-cold.log       # Server log (cold run)
│   │   ├── server-small-warm.log       # Server log (warm run)
│   │   ├── small-cold-response.json    # Raw API response
│   │   ├── small-cold-stats.json       # Extracted timing stats
│   │   ├── small-warm-response.json
│   │   ├── small-warm-stats.json
│   │   ├── small-result.json           # Cold vs warm comparison
│   │   ├── summary.json               # All sizes aggregated
│   │   └── summary.md                 # Human-readable table
│   └── summary.json / summary.md       # Aggregate across models
└── rocm/ ...
```

## Real-world CLIO performance

[CLIO](https://github.com/SyntheticAutonomicMind/CLIO) sends 18-30K tokens of system prompt, tool definitions, and prior conversation context on every API call. Without caching, every turn would re-evaluate the whole prompt from scratch. The numbers below come from one real session to show what the cache actually buys in practice.

### Workload profile

A CLIO session alternates between tool call turns (the LLM picks a tool and emits a JSON call) and response turns (the LLM writes a user-visible message). Tool call generations are short (~30-150 tokens). Response generations are longer - commands, code, explanations.

The prompt for every turn is split into two regions:

- **Static prefix** (~18K tokens, identical across turns): system prompt, tool definitions, the initial user message, the turn-0 assistant message, and the turn-0 tool results.
- **Dynamic tail** (grows ~0 to ~12K tokens): tool results and assistant responses from subsequent turns.


Without caching, every turn re-evaluates both regions. The static prefix is the obvious target - it's identical across turns and dominates prompt size. The dynamic tail stabilizes once the conversation does, so most of it can be cached too once it's been seen once.

### Test scenario

CLIO running Qwen3.6-35B-A3B (MoE hybrid, Q4_K_XL, Vulkan) on an Ayaneo Flip KB. Single user prompt: *"Please evaluate this project and share your opinion of it."* The model runs the project autonomously for five tool-calling turns, then writes a final evaluation. Cold server start with no cache populated.

### Results: cold start (no cache)

Tools per turn: `file_operations`, `version_control`, `terminal_operations`. T5 has no tool calls - it writes the final evaluation directly.

| Turn | Prompt tokens | TTFT | Gen t/s | Notes |
|------|---------------|------|---------|-------|
| T0 | 18,762 | 183.4s | 16.1 | Full system prompt + tools + initial user message. 102 t/s prompt eval rate. |
| T1 | 7,001 | 84.6s | 15.1 | Tool result reads. Static prefix restored from in-memory checkpoint. |
| T2 | 7,613 | 91.7s | 15.0 | Tool result reads, deeper into the runner script. |
| T3 | 7,115 | 86.6s | 15.2 | Tool result reads, terminal commands. |
| T4 | 5,784 | 69.2s | 15.1 | Final follow-up reads. Model preparing to write. |
| T5 | 649 | 10.1s | 15.2 | Writes the project evaluation. |

TTFT = time to first token (server-side prompt evaluation per turn). Gen t/s = generation speed (gen time = turn duration minus prompt eval). Prompt eval rate after T0: 82-83 t/s - the in-memory checkpoint restores the ~18K-token static prefix on each subsequent turn, so only the dynamic tail needs fresh evaluation.

**Total: 11 minutes 55 seconds with cache, vs ~42 minutes estimated without any cache.** Without caching, every turn would process all 18-26K tokens at the T0 eval rate of 102 t/s - 3 to 5 minutes of prompt eval per turn on top of generation. The 6:35 of total prompt eval time was concentrated in those per-turn TTFTs; the remaining 5:20 was generation and tool execution.

### Results: warm restart (cache populated)

Same workload, same machine. The warm run does NOT restart the OS or clear the cache directory - just the server, with the SSD cache and system prompt cache already on disk from the cold run.

| Turn | Cold TTFT | Cold prompt tok | Warm TTFT | Warm prompt tok | Speedup |
|------|-----------|-----------------|-----------|-----------------|---------|
| T0 | 183.4s | 18,762 | 3.4s | 101 | 54x |
| T1 | 84.6s | 7,001 | 19.5s | 1,555 | 4.3x |
| T2 | 91.7s | 7,613 | 25.0s | 1,947 | 3.7x |
| T3 | 86.6s | 7,115 | 23.1s | 1,767 | 3.7x |
| T4 | 69.2s | 5,784 | 64.5s | 5,380 | 1.1x |
| T5 | 10.1s | 649 | 43.0s | 3,044 | 0.2x |
| **Total** | **11:55 (715s)** | | **5:44 (344s)** | | **2.1x** |

#### Where the speedup comes from

**T0 (54x)** - the [system prompt cache](#system-prompt-cache) shines here. Cold T0 evaluates the full 18,762-token prompt from scratch. Warm T0 evaluates only 101 tokens because the 18,661-token system prompt is restored from the global cross-conversation cache. 180 seconds saved on the very first request of every server restart.

**T1-T3 (3.7-4.3x)** - the [SSD checkpoint layers](#search-strategy) restore 80-90% of the conversation context. The remaining 1,500-2,000 tokens to evaluate is just the new tool result plus the assistant's tool-call response. Speedup degrades from 4.3x to 3.7x as the conversation tail grows because each turn's divergent content is slightly larger than the last.

**T4 (1.1x)** - the conversation has nearly stabilized. The new tool result is similar in size to the prior turn's, so the checkpoint covers most of it. Caching and no-caching converge as the per-turn delta stops shrinking.

**T5 (0.2x)** - inversion. Warm T5 is *slower* than cold T5. The warm run generated 3,044 tokens of fresh content (a longer final evaluation), the cold run only 649 (a short one). Generation time dominates this turn, and the warm model wrote more. This shape shows up in any conversation where the final response branches late.

### Takeaways

- **The T0 speedup is the highest-value win.** 54x on the first request of every server restart, with no warmup needed. The system prompt cache turns "cold" server starts from a multi-minute wait into a sub-5-second response.
- **Per-turn speedup decays as conversations diverge.** Repeated reads of the same files cache well; a fresh exploration branch does not. SSD cache hit rate stays high on focused work and drops on divergent exploration.
- **Generation time is the floor.** Once prompt eval drops below generation time, more cache hits don't help - the model still has to write the response. The 0.2x on T5 is a feature, not a bug: it means caching isn't lying about its wins.
- **Local inference trades latency for privacy and cost.** Cloud-hosted models evaluate prompts in seconds on GPU clusters; the local model with SSD cache runs 10-183 seconds per turn. The tradeoff is no API keys, no per-token cost, no network dependency - usable offline.

The 2.1x total speedup is conservative. Longer sessions with more repeated tool calls (the common case in real agentic work) keep the SSD cache hit rate high and push the overall speedup toward 5-10x.

## What CachyLLama adds

CachyLLama is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
maintained as a standalone repo
([fewtarius/CachyLLama](https://github.com/fewtarius/CachyLLama)). All
custom changes - performance work, agentic workflow tuning, AMD APU
optimizations - are committed directly to that repo's git history. The
full design lives in [KV cache](#kv-cache), [User
isolation](#user-isolation), and [MoE expert tracking](#moe-expert-tracking).
The high-level changes:

### SSD-backed KV cache

Persistent cross-session KV cache that survives server restarts. Hot/warm/cold tiering with automatic promotion and demotion keeps frequently-used conversation state in RAM while evicting stale entries to disk. Per-conversation ring buffer prevents unbounded disk growth. Three-tier search (same-conversation, shared-prefix, cold-start token prefix) with chain/safe phases for cross-conversation safety. Kernel readahead overlaps SSD I/O with CPU work. Checkpoint overflow prevention handles cases where the saved state covers more tokens than the current task needs. Conversation hash and model compatibility hash prevent mismatched checkpoint restoration. Per-conversation directories (`kv-cache/{conv_hash}/`) let multiple chat threads run in parallel without interference. MLA model support for DeepSeek2/DeepSeek3.

CLI flags: `--cache-ssd`, `--cache-ssd-checkpoints`, `--cache-ssd-hot-window`, `--cache-ssd-warm-window`, `--cache-ssd-max-cold`, `--cache-ssd-page-size`, `--cache-ssd-max-conversations`, `--cache-ssd-hot-ram`, `--cache-ssd-warm-ram`, `--cache-ssd-system-prompts`, `--cache-ssd-system-max-days`.

### System prompt cache

Global cross-conversation cache for the system section of any prompt. First eval writes the state, subsequent requests skip the system prompt re-eval entirely. Default: 8 entries per model, 30 days unused before expiry. See the [System prompt cache](#system-prompt-cache) section for the full design including the per-position recurrent state handling for hybrid MoE/SSM models.

### Hybrid MoE support (Qwen3.5/3.6, GLM-4.7, Gemma 4)

Hybrid architectures mix attention and recurrent (Mamba) layers, which need different checkpoint handling than dense transformers. CachyLLama adds the right primitives for this:

- **KV cache shifting**: Hybrid models need different position tracking than dense models - pos_min/pos_max don't capture recurrent state coverage
- **Checkpoint erasure**: When conversation content diverges, only attention cells are cleared, preserving recurrent state for reuse
- **Checkpoint overflow prevention**: Same-conversation checkpoints accepted regardless of size (recurrent state is content-accurate), cross-conversation oversized checkpoints skipped at search
- **seq_rm_attn_only**: New API that clears attention KV entries without disturbing recurrent state
- **QWEN35MOE architecture filter**: Correctly identifies attention vs. recurrent layers
- **Checkpoint search condition**: Checkpoints are accepted only when
  `n_tokens >= n_past` (covers the prompt prefix) and `n_tokens <
  task_n_tokens` (fits within the current task). The first guard rejects
  divergent checkpoints whose stored state is shorter than the shared
  prefix; the second rejects checkpoints larger than the new task, which
  would leave no tokens to decode.

### User isolation

- **Per-user concurrency cap**: `--max-concurrent-per-user N` limits simultaneous slots per user. Returns HTTP 429 when the cap is hit.
- **User-scoped KV cache**: `user_id` routes checkpoints to `u/` namespace on disk, preventing cross-user cache contamination
- **Slot affinity**: Slot allocation prefers slots already owned by the requesting user for cache locality
- **Request threading**: `user_id` is threaded from the HTTP request body through `server_task` to slot allocation and cache routing

### MoE expert activation tracking

- **GET /expert-stats**: Per-layer expert activation counts, frequencies, and token counts
- **POST /expert-tracking**: Enable/disable tracking and reset counters
- **C API**: `llama_expert_tracking_enable()`, `llama_expert_stats_get()`, `llama_model_n_expert()`, `llama_model_n_expert_used()`
- Reads `ffn_moe_argsort` tensors from the compute graph after each decode to track which experts are activated per token

### Cache optimizations

- **Scoring-based prompt cache eviction**: Replaced FIFO eviction with scoring by age, size, and task token overlap. Conversations with long common prefixes stay cached longer
- **Text context on cache divergence**: Debug logging shows the actual tokens where cache diverged, making prompt engineering and tool output debugging tractable
- **Checkpoint eviction under memory pressure**: Automatically frees checkpoints when KV cache hits capacity limits
- **Conversation-aware checkpoint matching**: Uses model config validation to prevent mismatched checkpoint restoration

### Infrastructure

- **CLIO integration**: [CLIO](https://github.com/SyntheticAutonomicMind/CLIO) serializes tool definitions with deterministic JSON key ordering and reuses conversation state to maximize cache hits across agentic turns. System prompts, tool descriptions, and compressed context sent on every API call are cached and persisted to disk.
- **Auto-mlock tuning**: `llama-run.sh` compares model size against `RLIMIT_MEMLOCK` and disables `--mlock` when the limit is too small, eliminating startup warnings
- **SSD cache defaults**: Enabled by default for all non-SSM models in `llama-run.sh`. The `--cache-ssd-max-conversations` flag (default: 16) controls how many conversation directories are tracked simultaneously.
- **CPU ISA auto-detection**: `detect-gpu.sh` reads `/proc/cpuinfo` and generates optimal cmake flags for the detected CPU (AVX-512 BF16 on Zen 4, AVX2 on Zen 3, etc.), so AVX-512 code paths are enabled on hardware that supports them.

## Structure

```
├── llama-run.sh              # Main entry point
├── CachyLLama/               # Submodule - our fork of llama.cpp
├── scripts/
│   ├── rebuild.sh            # Build script (Vulkan default, optional ROCm)
│   ├── env.sh                # Environment setup (source before using tools)
│   ├── detect-gpu.sh         # GPU/APU and CPU ISA auto-detection library
│   ├── benchmark.sh          # Prompt cache performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
├── src/
│   ├── cachy-llama-rocm/     # ROCm build output + build.sh
│   ├── cachy-llama-vulkan/   # Vulkan build output + build.sh
│   └── cachy-llama-metal/    # Metal build output + build.sh (macOS)
├── deps/                     # ROCm SDK (downloaded by rebuild.sh)
├── models/                   # GGUF files
├── kv-cache/                 # SSD-backed KV cache (per-conversation directories)
├── scratch/                  # Transient working files (benchmark source text)
└── benchmarks/               # Benchmark results with full server logs
```

See [AGENTS.md](AGENTS.md) for the technical reference (directory structure, build commands, code style).

## License

Source code: [GPL-3.0-or-later](LICENSE)
Documentation: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

llama.cpp is MIT-licensed. ROCm components carry AMD's license.
