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

### Two-session workload

Two CLIO sessions against Qwen3.6-35B-A3B (MoE hybrid, Q4_K_XL, Vulkan backend, Ayaneo Flip KB):

- **Session 1** - Fresh server start. Read the README, investigated a code issue, committed a fix. 6 turns, 335s total.
- **Session 2** - Server restart with existing SSD cache. Reviewed benchmark results, analyzed cache behavior, edited documentation. 28 turns, 179s (first workflow) + 503s (second workflow).

#### Turn breakdown

| Turn | Tokens In | Tokens Out | Duration | ttft | tps | Cache Source |
|------|-----------|------------|----------|------|-----|-------------|
| **Session 1 (cold start)** |
| T0 - tool call | 19,967 | 139 | 175.0s | — | — | none (full eval) |
| T1 - tool call | 25,347 | 111 | 60.5s | — | — | in-memory ckpt |
| T2 - tool call | 25,813 | 134 | 14.4s | — | — | in-memory ckpt |
| T3 - tool call | 26,756 | 80 | 16.8s | — | — | in-memory ckpt |
| T4 - tool call | 25,720 | 80 | 8.3s | — | — | in-memory ckpt |
| T5 - response | 27,981 | 756 | 58.7s | 25.75s | 12.9 | in-memory ckpt |
| **Session 2 (SSD cold restore)** |
| T0 - tool call | 28,572 | 11 | 17.4s | 11.45s | 0.6 | SSD restore |
| T1 - tool call | 29,807 | 173 | 25.2s | — | — | SSD restore |
| T2 - tool call | 29,452 | 11 | 22.0s | 15.49s | 0.5 | SSD restore |
| T3 - tool call | 31,735 | 175 | 33.6s | — | — | SSD restore |
| T4 - tool call | 31,019 | 79 | 20.9s | — | — | SSD restore |
| T5 - tool call | 31,680 | 133 | 17.3s | — | — | SSD restore |
| T6 - response | 28,265 | 727 | 40.7s | 8.20s | 17.9 | SSD restore |

Tool call TTFT is .s because the full response arrives in one chunk - the model streams the tool call JSON and it's all received before the first measurable break. Response TTFT is the wait before streaming begins.

### What the cache saves

The static 20K-token prefix is the same in every turn. Without caching, it would be re-evaluated from scratch each time:

| Without cache | With cache |
|--------------|-----------|
| Session 1 first turn: 20K tokens cold eval (~175s) | Session 1 first turn: 20K cold eval (~175s) |
| Session 1 subsequent: 20K tokens re-evaluated each turn (~100-200s/turn) | Session 1 subsequent: cached, ~0-5K new tokens evaluated (~x-x/s/turn) |
| Session 2 first turn: 20K tokens cold eval again (~175s) | Session 2 first turn: 20K restored from SSD (~12s for ~8K new tokens) |
| Session 2 subsequent: same 20K re-evaluated | Session 2 subsequent: cached, same as session 1 |

**Net impact on these two sessions:**

- Session 1: avoided 5 cold re-evaluations of the static prefix. The 175s cold turn was followed by turns averaging ~23s for tool calls and ~59s for the response. Same workload without caching: every turn 175s+, totaling ~1050s vs 335s actual. **3x faster.**
- Session 2: avoided a full cold restart. The first turn took 17s instead of 175s. The 28-turn workload shows tool call turns averaging 10-34s and response turns 40s. Same workload without caching: every turn 175s+ for prompt eval alone, totaling ~4900s vs 179s. **27x faster.**

These speedups are smaller than the benchmark numbers because CLIO's prompt genuinely grows each turn (new conversation messages, updated tool counts). The cache restores the static prefix perfectly - but there's still 5-12K tokens of new content to evaluate each turn. The benchmark uses identical prompts, so the entire prompt is cached.

### Real-world cache behavior

The server log shows exactly where the cache saves time. From session 2's first turn:

```
slot update_slots: cache prefix divergence at token 20044: slot_token=271 input_token=198
SSD cache: within-conv match checkpoint 18 conv=c8074e0e67da355d n_tokens=28568 lcp=4096
slot update_slots: restored SSD checkpoint (n_tokens=28568)
slot update_slots: prompt processing done, n_tokens = 29452, batch.n_tokens = 4
```

The prompt diverges at token 20,044 (CLIO's tool usage stats differ between sessions). The cache finds checkpoint 18 at 28,568 tokens, restores it, and re-evaluates only ~900 tokens (29,452 total - 28,568 cached = 884 new). The first 28,568 tokens are the static system prompt, tool definitions, and compressed context - restored from disk in under a second.

Each subsequent turn in the session sees similar behavior: divergence at ~19,872-20,xx, cache restore of ~28K-30K tokens, and ~500-2,000 new tokens to evaluate. The dynamic portion (conversation history) is the only thing that changes between turns.

## Improvements over upstream

This fork maintains patches on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) that improve performance of agentic AI workloads with hybrid MoE models on AMD APU hardware.

### SSD-backed KV cache

Persistent cross-session KV cache that survives server restarts. Hot/warm/cold tiering with automatic promotion and demotion keeps frequently-used conversation state in RAM while evicting stale entries to disk.

- **Hot tier**: Recently-used checkpoints kept in RAM for instant restore. Demoted to warm after 2 inactive turns.
- **Warm tier**: Checkpoints from previous sessions, kept in RAM until memory pressure forces demotion. Demoted to cold after 4 inactive turns.
- **Cold tier**: On-disk checkpoints with token prefixes for cross-session matching
- **Ring buffer eviction**: Per-conversation ring buffer prevents unbounded disk growth. Oldest checkpoints are evicted when space is needed.
- **Three-tier search**: Same-conversation match by conversation hash, shared-prefix match by n_past, and cold-start token prefix comparison with chain/safe phases
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

### Cache optimizations

- **Scoring-based prompt cache eviction**: Replaced FIFO eviction with scoring by age, size, and task token overlap. Conversations with long common prefixes stay cached longer
- **Text context on cache divergence**: Debug logging shows the actual tokens where cache diverged, making prompt engineering and tool output debugging tractable
- **Checkpoint eviction under memory pressure**: Automatically frees checkpoints when KV cache hits capacity limits
- **Conversation-aware checkpoint matching**: Uses model config validation to prevent mismatched checkpoint restoration

### Infrastructure

- **CLIO integration**: [CLIO](https://github.com/SyntheticAutonomicMind/CLIO) serializes tool definitions with deterministic JSON key ordering and reuses conversation state to maximize cache hits across agentic turns. System prompts, tool descriptions, and compressed context sent on every API call are cached and persisted to disk.
- **Auto-mlock tuning**: `llama-run.sh` compares model size against `RLIMIT_MEMLOCK` and disables `--mlock` when the limit is too small, eliminating startup warnings
- **SSD cache defaults**: Enabled by default for all non-SSM models in `llama-run.sh`. The `--cache-ssd-max-conversations` flag (default: 16) controls how many conversation directories are tracked simultaneously.

## Structure

```
├── llama-run.sh              # Main entry point
├── llama.cpp/                # Submodule - ggml-org/llama.cpp
├── scripts/
│   ├── rebuild.sh            # Build script (Vulkan default, optional ROCm)
│   ├── env.sh                # Environment setup (source before using tools)
│   ├── detect-gpu.sh         # GPU/APU auto-detection library
│   ├── benchmark.sh          # Prompt cache performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
├── src/
│   ├── llama-cpp-rocm/       # ROCm build output + build.sh
│   └── llama-cpp-vulkan/     # Vulkan build output + build.sh
├── patches/                  # Patches applied to llama.cpp during build
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
