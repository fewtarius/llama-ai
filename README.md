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

Every API call in an agentic workflow sends static content: system prompt, tool definitions, compressed conversation context. Without caching, this content is re-evaluated from scratch on every single call. A 15K-token prompt means 2-4 minutes before the model starts responding. With SSD cache, the same prompt evaluates in 1-4 seconds.

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

The KV cache (attention state) and recurrent state (for hybrid MoE models) are restored from the checkpoint. Only tokens beyond the checkpoint's coverage need evaluation. A 15K-token prompt might need just a handful of new tokens evaluated - the rest is restored from disk in under a second.

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

Tested on Ayaneo Flip KB (7840U / 780M / 32GB / Vulkan). 128 output tokens, ctx 32768, all GPU layers.

#### GLM-4.7-Flash (Q4_K_M, 14B dense)

| Size | Tokens | Cold TTFT | Warm TTFT | Speedup | Gen TPS |
|------|--------|-----------|-----------|---------|---------|
| Small | ~1145 | 12.4s | 0.4s | 32.4x | 22.1 |
| Medium | ~5237 | 116.5s | 1.3s | 89.6x | 15.1 |
| Large | ~15.5K | 800.9s (13.3min) | 4.0s | 200.5x | 8.4 |

Cold prompt eval: 19-93 t/s. Warm: 2994-3878 t/s. Cached: 15485/15489 tokens at large size.

#### Gemma 4 26B (Q5_K_M, 26B dense)

| Size | Tokens | Cold TTFT | Warm TTFT | Speedup | Gen TPS |
|------|--------|-----------|-----------|---------|---------|
| Small | ~1413 | 10.5s | 1.2s | 8.7x | 17.4 |
| Medium | ~6083 | 49.1s | 1.6s | 31.5x | 16.9 |
| Large | ~17.3K | 191.7s (3.2min) | 2.4s | 81.2x | 15.8 |

Cold prompt eval: 90-134 t/s. Warm: 1164-7350 t/s. Cached: 17343/17347 tokens at large size.

#### Qwen3.6-35B (Q4_K_XL, 35B MoE, hybrid)

| Size | Tokens | Cold TTFT | Warm TTFT | Speedup | Gen TPS |
|------|--------|-----------|-----------|---------|---------|
| Small | ~1243 | 8.8s | 0.5s | 17.9x | 21.3 |
| Medium | ~5409 | 39.6s | 0.7s | 53.7x | 20.8 |
| Large | ~15.7K | 128.8s (2.1min) | 1.5s | 87.9x | 19.2 |

Cold prompt eval: 122-142 t/s. Warm: 2542-10731 t/s. Cached: 15717/15721 tokens at large size.
35B parameters with only 3B active - the fastest model tested on the Flip. The SSD cache restores both attention KV state and recurrent state from disk. Only 4 new tokens need evaluation at large size.

#### Summary

All models on Ayaneo Flip KB (7840U / 780M / 32GB / Vulkan):

| Model | Params | Large cold | Large warm | Speedup | Gen TPS | Type |
|-------|--------|------------|------------|---------|---------|------|
| GLM-4.7-Flash | 14B | 800.9s (13.3min) | 4.0s | 200.5x | 8.4 | Dense |
| Gemma 4 26B | 26B | 191.7s (3.2min) | 2.4s | 81.2x | 15.8 | Dense |
| Qwen3.6-35B | 35B | 128.8s (2.1min) | 1.5s | 87.9x | 19.2 | MoE hybrid |

Generation speed (t/s) is unaffected by caching - the speedup is entirely in prompt evaluation. What caching changes is whether you wait 2-13 minutes or 1-4 seconds before the model starts responding.

Full benchmark data (server logs, API responses, timing stats): [`benchmarks/20260530-1525/`](benchmarks/20260530-1525/)

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

This cache was built for [CLIO](https://github.com/SyntheticAutonomicMind/CLIO), an AI coding assistant that sends 20-32K tokens of system prompt, tool definitions, and compressed conversation context on every API call. Without caching, every turn would re-evaluate all 20K+ tokens from scratch.

### Workload profile

A CLIO session consists of alternating tool call turns (the LLM decides what tool to run) and response turns (the LLM generates a user-visible message). Tool call turns are short - the model outputs a tool call JSON (~30-150 tokens). Response turns are longer - the model generates commands, code, and explanations.

Every turn includes the same static prefix: system prompt, tool definitions, project context. As the conversation grows, compressed summaries of earlier messages are appended. The static portion is ~20K tokens; the dynamic conversation portion grows from ~5K to ~12K.

### CLIO agentic workload

A CLIO session against Qwen3.6-35B-A3B (MoE hybrid, Q4_K_XL, Vulkan backend, Ayaneo Flip KB). The session reviewed a project, investigated code, and committed changes - 14 turns, ~21 minutes total.

#### Turn breakdown

| Turn | Tokens In | Cached | Cache% | New Toks | Prompt Eval | Speedup | TTFT |
|------|-----------|-------|--------|----------|-------------|---------|------|
| T0 - cold start | 17,802 | 0 | 0% | 17,802 | 156.1s | 1.0x | 158.5s |
| T1 - tool call | 19,753 | 17,887 | 91% | 1,866 | 21.9s | 7.9x | 23.8s |
| T2 - tool call | 19,976 | 17,691 | 89% | 2,285 | 26.9s | 6.5x | 29.1s |
| T3 - tool call | 19,985 | 17,897 | 90% | 2,088 | 25.4s | 6.9x | 27.4s |
| T4 - tool call | 26,346 | 20,311 | 77% | 6,035 | 70.0s | 3.3x | 71.8s |
| T5 - tool call | 29,010 | 17,905 | 62% | 11,105 | 129.2s | 2.0x | - |
| T6 - tool call | 25,374 | 17,914 | 71% | 7,460 | 85.3s | 2.6x | - |
| T7 - tool call | 29,067 | 25,574 | 88% | 3,493 | 45.2s | 5.6x | - |
| T8 - tool call | 27,365 | 17,955 | 66% | 9,410 | 109.5s | 2.2x | - |
| T9 - tool call | 30,491 | 17,964 | 59% | 12,527 | 148.2s | 1.8x | - |
| T10 - tool call | 30,933 | 30,583 | 99% | 350 | 7.4s | 36.6x | - |
| T11 - tool call | 28,133 | 17,973 | 64% | 10,160 | 119.2s | 2.1x | - |
| T12 - tool call | 28,652 | 28,220 | 98% | 432 | 9.6s | 26.1x | - |
| T13 - response | 29,392 | 28,781 | 98% | 611 | 11.7s | 22.1x | 40.7s |

Cache% = cached tokens / total tokens. Speedup = (total tokens * cold eval rate) / actual eval time. TTFT is blank for tool-call turns where the response arrives as a single chunk.

Generation speed: 17.5-18.9 t/s across all turns (unaffected by caching).

#### Cache behavior

The in-memory checkpoint system restores ~17,900 tokens of static prefix (system prompt + tool definitions) on every turn. This is the common content shared across all turns of a CLIO session. Only the dynamic portion (conversation history, tool results) needs re-evaluation.

Two patterns emerge:

1. **High cache hit (88-99%)**: Turns where the prompt is similar in size to the previous turn. The checkpoint from the previous turn covers most of the new prompt. Prompt eval drops to 7-26 seconds (7-36x faster than cold).

2. **Moderate cache hit (59-71%)**: Turns where the prompt grew significantly (new tool results, context expansion). The checkpoint covers the static prefix but the dynamic portion needs full evaluation. Still 1.8-3.3x faster than cold.

The best results (T10, T12, T13) show 22-37x speedup when the prompt is nearly identical to the previous turn's checkpoint. Even the worst case (T9, 59% cache hit) is still 1.8x faster than cold evaluation.

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