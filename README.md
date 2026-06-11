# llama-ai

Local LLM inference on AMD APU hardware using [llama.cpp](https://github.com/ggml-org/llama.cpp). Self-contained - no system ROCm install required. Vulkan (RADV) is the default backend for best stability on RDNA3 iGPUs.

## Why

The goal is reasonably-performing agentic AI development on an [Ayaneo Flip KB](https://ayaneo.com/product/AYANEO-FLIP-KB) (7840U / 32GB) handheld - usable when there is no network. No API keys, no per-token costs, no cloud dependency. Cached state survives reboots and power outages (the Flip has a battery).

[CLIO](https://github.com/SyntheticAutonomicMind/CLIO) is optimized for this implementation. It serializes tool definitions with deterministic JSON key ordering and reuses conversation state to maximize cache hits across agentic turns. System prompts, tool descriptions, and compressed context - the static content sent on every API call - are cached and persisted to disk so they're available immediately on the next request.

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

To build with ROCm support (optional, has stability issues on some architectures):

```bash
./scripts/rebuild.sh --both    # Vulkan + ROCm
./scripts/rebuild.sh --rocm    # ROCm only
```

## GPU memory

AMD APUs share system RAM with the GPU. Use `apply-ttm-kernel-params.sh` to configure GTT:

```bash
# Set GTT to 18GB (total GPU memory: 6GB VRAM + 18GB GTT = 24GB)
sudo ./scripts/apply-ttm-kernel-params.sh 18
sudo reboot
```

Writes kernel parameters (`amdgpu.gttsize`, `amdgpu.vis_vramlimit`, `ttm.pages_limit`) to your bootloader config. Also calls `amd-smi set -G` as a runtime hint, but kernel parameters are the authoritative method that persists across reboots.

Supports GRUB (SteamFork 3.7) and systemd-boot (SteamFork 3.8+). Tested on SteamFork - may not work with other distributions.

GTT size defaults to auto-detected value based on total system RAM (reserves 6GB for OS). Override with the first argument or `LLAMA_GTT_SIZE` env var.

Verify after reboot:
```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E "amdgpu|ttm"
```

## GPU detection

Auto-detects AMD GPU via PCI device ID and sets `HSA_OVERRIDE_GFX_VERSION` for ROCm.

Supported: Cezanne (5800H), Phoenix (780M), Hawk Point (890M/780M), Strix Point (890M/880M), Strix Halo, Sephiroth, Rembrandt (680M/660M), Mendocino (610M), Renoir, Lucienne. Falls back to `amd-smi` for authoritative detection when PCI IDs are ambiguous (e.g. Cezanne and Van Gogh share the same PCI ID). To add your device, edit the `GPU_MAP` in `scripts/detect-gpu.sh`.

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

## How it works

Models are auto-profiled based on filename characteristics. MoE models get checkpoint strategies and reasoning format; SSM/Mamba models get context-shift disabled; large dense models get optimized batch sizes. Profiles are assigned dynamically - no hard-coded model names.

SSD-backed KV cache persists conversation state across server restarts. Enabled by default for all non-SSM models - the cache directory is `kv-cache/`. When available, ROCm is auto-detected as a secondary backend option.

## Benchmarking

The bottleneck in agentic AI isn't generation speed (the model produces tokens as fast as the GPU allows). The bottleneck is **prompt evaluation** - reprocessing the entire prompt before the model can generate its first token.

Every API call in an agentic workflow sends static content: system prompt, tool definitions, prior conversation context. Without caching, this content is re-evaluated from scratch on every single call. An 18-30K-token prompt means it could be several minutes before the model starts responding on an APU like the 780M. With SSD cache and a 17,800-token prefix hit, only the divergent tail of the prompt is evaluated - typically a few seconds when only the latest tool result is new.

### How the cache works

The SSD-backed KV cache has three tiers with automatic promotion and demotion:

- **Hot tier** - Checkpoints from the current session, kept in RAM. Instant restore when the same conversation continues. After 2 turns of inactivity, hot checkpoints are demoted to warm.
- **Warm tier** - Checkpoints from previous sessions in the same server run. In RAM until memory pressure forces demotion to cold. After 4 turns of inactivity, warm checkpoints are demoted to cold.
- **Cold tier** - On-disk checkpoints with token prefixes. Survives server restarts. Each conversation gets up to the ring buffer limit of cold checkpoints on disk. When the limit is exceeded, the oldest cold checkpoint is deleted. Up to 16 conversations are tracked simultaneously (configurable with `--cache-ssd-max-conversations`).

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

Tested on Ayaneo Flip KB (7840U / 780M / 32GB / Vulkan). 128 output tokens, ctx 32768, all GPU layers. All three sizes use SSD cold cache (server restart with checkpoint restored from disk).

#### GLM-4.7-Flash (Q4_K_M, 14B dense)

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
35B parameters with only 3B active keeps the eval rate high. The SSD cache restores both attention KV state and recurrent state from disk — the hybrid architecture's Mamba layers are checkpoint-aware and restore correctly across restarts.

#### Summary

All models on Ayaneo Flip KB (7840U / 780M / 32GB / Vulkan):

| Model | Params | Active | Large cold | Large warm | Speedup | Gen TPS |
|-------|--------|--------|------------|------------|---------|---------|
| GLM-4.7-Flash | 14B | 14B | 467.6s (7.8min) | 2.7s | 174.1x | 5.7 |
| Gemma 4 26B | 26B | 4B | 130.9s (2.2min) | 1.4s | 92.9x | 13.8 |
| Qwen3.6-35B | 35B | 3B | 143.1s (2.4min) | 1.0s | 144.5x | 18.6 |

Generation speed (t/s) is unaffected by caching — the speedup is entirely in prompt evaluation. What caching changes is whether you wait 3-5 minutes or 1-4 seconds before the model starts responding. MoE models trade parameter count for active headroom: Qwen loads 35B weights but only evaluates 3B per token, giving it the best generation speed of the three.

Full benchmark data (server logs, API responses, timing stats): [`benchmarks/20260611-0656/`](benchmarks/20260611-0656/)

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

This cache was built for [CLIO](https://github.com/SyntheticAutonomicMind/CLIO), an AI coding assistant that sends 18-30K tokens of system prompt, tool definitions, and prior conversation context on every API call. Without caching, every turn would re-evaluate all 18K+ tokens from scratch.

### Workload profile

A CLIO session consists of alternating tool call turns (the LLM decides what tool to run) and response turns (the LLM generates a user-visible message). Tool call turns are short - the model outputs a tool call JSON (~30-150 tokens). Response turns are longer - the model generates commands, code, and explanations.

Every turn includes the same static prefix: system prompt, tool definitions, the initial user message, and all prior conversation messages. As the conversation grows, new tool results get appended to the end. The static prefix that never changes across turns is ~18K tokens (system prompt + tool definitions + initial user message + assistant turn 0 + tool results 0). The dynamic tail grows from ~0 to ~12K tokens as new tool results accumulate.

### Agentic workflow walkthrough

A single prompt — "Please evaluate this project and share your opinion of it." — sent to CLIO running on Qwen3.6-35B-A3B (MoE hybrid, Q4_K_XL, Vulkan, Ayaneo Flip KB). Ten turns, 18 minutes 10 seconds.

The model explores the project autonomously: it lists directories, reads source files, checks git history, and gradually builds understanding before writing a detailed evaluation. Each turn sends the full conversation context (18-30K tokens) to the API. The SSD-backed KV cache determines how much of that context needs fresh evaluation.

#### Turn-by-turn

Tools per turn: `list_dir`, `read_file`, `version_control/log`. T9 has no tool calls — it writes the final evaluation message directly.

| Turn | Model action | Tokens | Cached | Cache% | TTFT | Gen t/s |
|------|-------------|--------|--------|--------|------|---------|
| T0 | List project root, read `llama-run.sh` and `benchmark.sh` | 17,984 | 0 | 0% | 172s | 19.0 |
| T1 | Recursive directory listing, git log | 18,148 | 17,882 | 98.5% | 5.3s | 18.9 |
| T2 | Re-read `llama-run.sh` and `benchmark.sh`, list project root | 28,557 | 18,239 | 63.9% | 128s | 17.4 |
| T3 | Re-read `llama-run.sh`, list `scripts/` and `src/` | 28,565 | 18,029 | 63.1% | 132s | 17.4 |
| T4 | Read `detect-gpu.sh` | 23,429 | 18,037 | 77.0% | 66s | 17.6 |
| T5 | Re-read `detect-gpu.sh` | 23,462 | 18,045 | 76.9% | 68s | 17.5 |
| T6 | Read `llama-run.sh:200-400`, `detect-gpu.sh:1-100`, `benchmark.sh:1-100` | 23,471 | 18,078 | 77.0% | 68s | 17.6 |
| T7 | Read `detect-gpu.sh:1-100`, `rebuild.sh:1-80`, git log | 29,460 | 23,725 | 80.5% | 78s | 16.8 |
| T8 | Read `README.md:1-80`, `env.sh` | 29,340 | 18,086 | 61.6% | 144s | 16.9 |
| T9 | Write project evaluation (no tool calls) | 25,724 | 18,119 | 70.4% | 96s | 17.2 |

TTFT = time to first token (server-side prompt evaluation time per turn). Gen t/s = generation tokens per second (gen time = turn duration minus prompt eval). Cold eval rate at T0: 105 t/s. Cached eval rate for later turns: 74-82 t/s.

**Total: 18 minutes 10 seconds actual vs ~42 minutes estimated without any cache.**

#### What the model explored

Over nine tool-calling turns the model read six unique files — `llama-run.sh`, `benchmark.sh`, `detect-gpu.sh`, `rebuild.sh`, `env.sh`, `README.md` (with multiple line ranges and re-reads) — listed the project root, `scripts/`, and `src/`, and called `git log` twice. The exploration was broad rather than fixated: each turn targeted different files or different sections within those files.

The model's final evaluation called out five standout areas: the GPU detection system ("genuinely thorough," covering GCN5 through RDNA3.5 across 20+ device variants), the SSD cache work ("real systems-level expertise"), clean bash scripting (`set -euo pipefail`, `SCRIPT_DIR`/`PROJECT_ROOT`, color-coded logging), the multi-backend architecture (Vulkan default, ROCm optional, Metal for macOS), and the kernel parameter management for GTT/VRAM carveout (writing GRUB/systemd-boot config, verifying via `/proc/cmdline`).

#### How the cache performed

The ~18K-token static prefix — system prompt, tool definitions, the initial user message, and the turn-0 assistant/tool result exchange — is cached after T0 and never re-evaluated. Every turn reuses these tokens from an in-memory or SSD checkpoint. The LCP between consecutive turns is consistently 17,882-23,725 tokens depending on how much of the prior conversation the next turn's input shares.

Cache hit rates range from 61.6% to 98.5% depending on how much the conversation has diverged:

- **Near-perfect (98.5%, T1):** T1's input starts with the same 17,882 tokens T0 ended with — system prompt, tools, user message, T0's assistant turn, T0's tool results. Only 266 new tokens evaluated (the model's new tool call and the streaming chunk header). In-memory checkpoint restored in milliseconds.

- **Typical tool turns (63-77%, T2-T7):** The model explores different files and the conversation context grows organically. The static prefix is always cached; the dynamic conversation content varies with model choices. 5,392-11,254 new tokens evaluated per turn at 74-82 t/s.

- **Deep exploration (61.6%, T8):** The model's input at T8 diverges from the prior in-memory checkpoint at token 18,086 (T7 took a different path through the conversation). More tokens need fresh evaluation than other turns, but 18,086 are still restored from cache — 11,254 fresh tokens at 78 t/s.

- **Evaluation output (70.4%, T9):** The model writes a 632-token assessment with no tool calls. 18,119 tokens from cache, 7,605 fresh. The evaluation turn is generation-bound (37s of output at 17.2 t/s) rather than evaluation-bound.

#### What the numbers mean

Without caching, every turn would process all 18-30K tokens from scratch at 105 t/s — 3 to 5 minutes per turn. The SSD cache eliminates evaluation of the static prefix entirely and captures much of the dynamic conversation as it stabilizes. Real agentic workloads benefit from this daily: a 10-turn exploration session drops from ~42 minutes (estimated cold) to 18 minutes (cached).

The tradeoff: cloud-hosted models evaluate prompts in seconds on GPU clusters. The local model with SSD cache achieves per-turn latency of 5-144 seconds. Local inference is private, offline-capable, and has no per-token cost.

## User isolation

Multi-tenant deployments need isolation between users sharing the same server. This fork adds three dimensions of isolation:

### Identity

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

### KV cache routing

When `user_id` is present, the SSD page manager routes checkpoints to a separate `u/` namespace on disk:

```
{ssd_path}/{hash_hex}/    # anonymous (conv_hash)
{ssd_path}/u/{hash_hex}/  # user-scoped (fnv1a(user_id))
```

Cross-user lookup is disabled for user-scoped requests. A user can only access their own cached state, never another user's directory.

### Scheduling isolation

`--max-concurrent-per-user N` caps the number of simultaneous slots a single user_id can occupy. When the cap is hit, the server returns HTTP 429 with a `rate_limit_error` type:

```json
{
  "error": {
    "code": 429,
    "message": "User 'tenant-42-user-7' has reached the concurrent request limit (2)",
    "type": "rate_limit_error"
  }
}
```

Slot allocation also prefers slots already owned by the requesting user (cache affinity). An empty slot (post-release) is fair game for any user.

Default: 0 (unlimited). Set to 1 for strict one-at-a-time, or 2-3 for concurrent with backpressure.

Design rationale: [`docs/development/user-isolation-design.md`](llama.cpp/docs/development/user-isolation-design.md)

## MoE expert tracking

MoE models (Qwen3.5/3.6, DeepSeek, Mixtral) activate only a subset of experts per token. This fork adds real-time expert activation tracking via two HTTP endpoints:

### GET /expert-stats

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

### POST /expert-tracking

Enable/disable tracking and optionally reset counters:

```json
{"enabled": true, "reset": true}
```

This is Phase 1 of the MoE expert tiering design - instrumentation only, no compute changes. Future phases will use this data to reorder experts for cache locality and offload cold experts to RAM/SSD.

## Improvements over upstream

This fork maintains patches on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) that improve performance of agentic AI workloads with hybrid MoE models on AMD APU hardware.

### SSD-backed KV cache

Persistent cross-session KV cache that survives server restarts. Hot/warm/cold tiering with automatic promotion and demotion keeps frequently-used conversation state in RAM while evicting stale entries to disk.

- **Hot tier**: Recently-used checkpoints kept in RAM for instant restore. Demoted to warm after 2 inactive turns.
- **Warm tier**: Checkpoints from previous sessions, kept in RAM until memory pressure forces demotion. Demoted to cold after 4 inactive turns.
- **Cold tier**: On-disk checkpoints with token prefixes for cross-session matching
- **Ring buffer eviction**: Per-conversation ring buffer prevents unbounded disk growth. Oldest checkpoints are evicted when space is needed.
- **Three-tier search**: Same-conversation match by conversation hash, shared-prefix match by n_past, and cold-start token prefix comparison with chain/safe phases
- **Kernel readahead**: `posix_fadvise(POSIX_FADV_WILLNEED)` on Linux, `readahead()` on macOS. Overlaps SSD I/O with CPU work for ~0.5-0.75s TTFT reduction on cold cache hits.
- **Checkpoint overflow prevention**: Same-conversation checkpoints are accepted regardless of size (recurrent state is content-accurate) and capped in the restore layer. Cross-conversation oversized checkpoints are skipped at the search layer. Prevents "no tokens to decode" crashes.
- **Turn-based tiering**: Checkpoints track turn activity across server restarts for accurate promotion/demotion
- **Cold start recovery**: On server restart, automatically searches SSD cache by token prefix match. Same-conversation checkpoints are restored even if larger than the current task - `n_past` is capped with overflow margin instead of falling through to full reprocessing.
- **Conversation-aware matching**: Checkpoints carry conversation hash and model compatibility hash. Mismatched checkpoints are rejected, so switching models or conversations doesn't corrupt cached state.
- **Per-conversation directories**: Each conversation gets its own directory (`kv-cache/{conv_hash}/`). Switching conversations doesn't corrupt cached state. Multiple independent chat threads operate in parallel without interference.
- **Smart eviction**: Scores checkpoints by age, size, and task overlap to preserve what's most useful
- **MLA model support**: DeepSeek2/DeepSeek3 MLA models get checkpoint support via `llama_model_is_mla()` detection

CLI flags: `--cache-ssd`, `--cache-ssd-checkpoints`, `--cache-ssd-hot-window`, `--cache-ssd-warm-window`, `--cache-ssd-max-cold`, `--cache-ssd-page-size`, `--cache-ssd-max-conversations`

### Hybrid MoE model fixes (Qwen3.5/3.6)

Hybrid models (Qwen3.5/3.6 MoE) combine transformer attention with recurrent state (like Mamba). Upstream checkpoint restore was broken for these architectures, causing silent KV cache exhaustion and `no tokens to decode` crashes after 2-3 conversation turns. 13 incremental fixes:

- **KV cache shifting**: Hybrid models need different position tracking than dense models - pos_min/pos_max don't capture recurrent state coverage
- **Checkpoint erasure**: When conversation content diverges, only attention cells are cleared, preserving recurrent state for reuse
- **Checkpoint overflow prevention**: Same-conversation checkpoints are accepted regardless of size (recurrent state is content-accurate) and capped in the restore layer. Cross-conversation oversized checkpoints are skipped at the search layer. Prevents the fatal `batch.n_tokens = 0` crash
- **seq_rm_attn_only**: New API that clears attention KV entries without disturbing recurrent state - critical for checkpoint restore correctness
- **QWEN35MOE architecture filter**: Correctly identifies which layers are attention vs. recurrent for state management
- **Checkpoint search condition**: Hybrid checkpoint restore uses `n_tokens <= n_past` to prevent restoring recurrent state from stale (diverged) conversation content. The previous `>=` condition allowed checkpoints past the cache divergence point, causing degraded output on multi-turn conversations

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
- **CPU ISA auto-detection**: `detect-gpu.sh` reads `/proc/cpuinfo` and generates optimal cmake flags for the detected CPU (AVX-512 BF16 on Zen 4, AVX2 on Zen 3, etc.). Previously, the Vulkan build was compiled with `GGML_NATIVE=OFF` and `GGML_AVX512=OFF`, leaving AVX-512 code paths disabled on hardware that supports them.

## Structure

```
├── llama-run.sh              # Main entry point
├── llama.cpp/                # Submodule - ggml-org/llama.cpp
├── scripts/
│   ├── rebuild.sh            # Build script (Vulkan default, optional ROCm)
│   ├── env.sh                # Environment setup (source before using tools)
│   ├── detect-gpu.sh         # GPU/APU and CPU ISA auto-detection library
│   ├── benchmark.sh          # Prompt cache performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
├── src/
│   ├── llama-cpp-rocm/       # ROCm build output + build.sh
│   └── llama-cpp-vulkan/     # Vulkan build output + build.sh
├── deps/                     # ROCm SDK (downloaded by rebuild.sh)
├── models/                   # GGUF files
├── kv-cache/                 # SSD-backed KV cache (per-conversation directories)
├── scratch/                  # Transient working files (benchmark source text)
└── benchmarks/               # Benchmark results with full server logs
```

## License

Source code: [GPL-3.0-or-later](LICENSE)
Documentation: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

llama.cpp is MIT-licensed. ROCm components carry AMD's license.
