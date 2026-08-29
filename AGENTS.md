# AGENTS.md

Technical reference for llama-ai development (the runner scripts, solver, GPU
detection, and build orchestration layer). For C/C++ engine development, see
[CachyLLama/AGENTS.md](CachyLLama/AGENTS.md). For project methodology (the
Unbroken Method, checkpoint workflow, session handoff), see
`.clio/instructions.md`.

---

## Project overview

llama-ai is the runner layer that sits on top of the CachyLLama inference engine.
It handles:

- GPU/CPU auto-detection (`scripts/detect-gpu.sh`)
- Build orchestration for Vulkan, ROCm, and Metal backends (`scripts/rebuild.sh`)
- Adaptive profile selection via an optimistic-first solver (`scripts/optimize.sh`)
- Server launch with the right flags (`llama-run.sh`)
- Benchmark harness (`scripts/benchmark.sh`)
- GPU memory configuration (`scripts/apply-ttm-kernel-params.sh`)
- Model download and integrity testing (`scripts/lib-discover-models.sh`,
  `tests/test-download-integrity.sh`)

- **Languages:** Bash (scripts, runner), Python (GGUF metadata reader, log analysis)
- **Build system:** CMake via `scripts/rebuild.sh`
- **Primary target:** Nimo Axis N161 (Strix Halo, Ryzen AI Max+ 395, gfx1151)
- **Secondary:** Ayaneo Flip KB (7840U / gfx1103), Minisforum UM580 (5800H / gfx90c)
- **License:** GPL-3.0-or-later (source), CC-BY-NC-SA-4.0 (documentation)

---

## Directory structure

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
│   ├── read_gguf_kv.py       # GGUF metadata reader (used by solver)
│   └── log_analyzer.py       # Benchmark log analysis
├── src/
│   ├── cachy-llama-vulkan/   # Vulkan build output + build.sh
│   ├── cachy-llama-rocm/     # ROCm build output + build.sh
│   └── cachy-llama-metal/    # Metal build output + build.sh (macOS)
├── deps/                     # ROCm SDK (downloaded by rebuild.sh, gitignored)
├── models/                   # GGUF files (gitignored)
├── kv-cache/                 # SSD-backed KV cache (gitignored)
├── scratch/                  # Transient working files (gitignored)
└── tests/
    └── test-download-integrity.sh  # Download cache and split-shard validation
```

---

## Build

```bash
# Full setup from fresh checkout (builds Vulkan backend by default)
./scripts/rebuild.sh

# Full rebuild with ROCm support
./scripts/rebuild.sh --rocm

# Build both backends
./scripts/rebuild.sh --both

# Full rebuild from scratch (cleans first)
./scripts/rebuild.sh --rebuild

# Build Metal on macOS (auto-detected)
./scripts/rebuild.sh
```

Build scripts in `src/cachy-llama-vulkan/build.sh` and
`src/cachy-llama-rocm/build.sh` reference `$PROJECT_ROOT/CachyLLama` for the
source. CachyLLama is maintained directly in its git history — no patches are
applied. See [CachyLLama/AGENTS.md](CachyLLama/AGENTS.md) for C++ development
conventions.

### Build flow

1. `scripts/rebuild.sh` detects the platform and selects the backend(s)
2. Downloads the ROCm SDK (if `--rocm` or `--both`) into `deps/`
3. Sources `scripts/env.sh` to set `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`
4. Sources `scripts/detect-gpu.sh` to get GPU/CPU info and cmake flags
5. Runs cmake with the right flags, including CPU ISA detection
6. Builds in `src/cachy-llama-*/build/`

---

## Environment

```bash
# Required before using Vulkan tools (default)
source scripts/env.sh vulkan

# Or for ROCm (optional — ROCm has stability issues on RDNA3)
source scripts/env.sh rocm
```

Sets `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`, `PATH`. On macOS, Metal is
auto-detected and no environment sourcing is needed.

---

## GPU detection

`scripts/detect-gpu.sh` is sourced by `llama-run.sh` and `rebuild.sh`. It
identifies your AMD GPU via PCI device ID and sets:

| Variable | Description |
|----------|-------------|
| `LLAMA_IS_STRIX_HALO` | `1` on Strix Halo (`1002:1586`), `0` elsewhere |
| `LLAMA_HARDWARE_TIER` | `halo` / `standard` / `handheld` (back-compat) |
| `LLAMA_APU_VRAM_GB` | VRAM carveout in GB |
| `LLAMA_TOTAL_RAM_GB` | Total system RAM in GB |
| `LLAMA_THREADS` | Optimal thread count (physical cores / 2 for batch) |
| `LLAMA_GFX_ARCH` | e.g. `gfx1151` (Strix Halo), `gfx1103` (7840U) |
| `LLAMA_GPU_NAME` | e.g. `Radeon 8060S`, `Radeon 780M` |
| `LLAMA_RECOMMENDED_GTT_GB` | Recommended GTT size in GB |

The script also detects CPU ISA level and exports cmake flags:

| CPU | ISA flags |
|-----|-----------|
| Zen 4 (Phoenix, Strix Halo) | `GGML_AVX512=ON GGML_AVX512_BF16=ON` |
| Zen 3 (7840U, 5800H) | `GGML_AVX2=ON` |
| Apple Silicon | `GGML_NATIVE=ON` |

### Adding a new GPU

Find your PCI ID:
```bash
lspci -nn | grep VGA
```

Add an entry to the `GPU_MAP` array in `scripts/detect-gpu.sh`:
```bash
GPU_MAP["1002:1586"]="11.5.1:gfx1151:Radeon 8060S"
```

Format: `"PCI_ID= GFX_VERSION:gfx_arch:GPU_NAME"`

### Overrides

```bash
# Environment
LLAMA_GFX_VERSION_OVERRIDE=11.0.3
LLAMA_IS_STRIX_HALO_OVERRIDE=1
LLAMA_HARDWARE_TIER_OVERRIDE=halo

# CLI
./llama-run.sh --model ... --hardware-tier halo
./llama-run.sh --model ... --is-strix-halo
./llama-run.sh --model ... --no-strix-halo
```

---

## The solver (`scripts/optimize.sh`)

`llama-run.sh` sources `scripts/optimize.sh` and calls `solve_optimal_config()`
to compute the best profile for a given model and hardware. The solver is
optimistic-first: it starts with the per-archetype (batch, ubatch) defaults
and the highest-quality configuration, then iterates a phase-1 scoring loop
across (strategy, ctx, KV, draft, batch, ubatch) candidates, picking the
first combo that fits the GPU budget. If nothing fits, phase 2 detunes
through the priority list below until something does.

### Phase 2 detune priority (after the phase-1 combination loop)

1. Reduce KV cache to q8_0 (~50% memory save, minor quality)
2. Reduce KV cache to q4_0 (~75% memory save, more quality)
3. Reduce NGL by 10% per step (CPU offload, small decode cost)
4. Reduce SSD hot/warm RAM by half (no prefill/decode impact)
5. Drop speculative draft model (some decode throughput loss)
6. Drop SSD cache entirely
7. Reduce ubatch by half, clamped at 512
8. Reduce ctx (cascading values 262144 -> 196608 -> 131072 -> 98304 -> 65536)

Each step has a "done" flag so it only applies once.

### Per-archetype (batch, ubatch) defaults

Driven by llama-bench sweeps on 17 model/hardware combinations (see
[SOLVER.md](SOLVER.md) for the data). Decode (tg) is memory-bandwidth bound
and varies <2% across the (batch, ubatch) range; prefill (pp) is what
tuning moves.

| Tier | Archetype | Size | ubatch | batch |
|------|-----------|------|--------|-------|
| halo | dense | any | 1024 | 4096 |
| halo | MoE small | <60 GB | 1024 | 2048 |
| halo | MoE large | 60-100 GB | 2048 | 8192 |
| halo | MoE huge | >=100 GB | 4096 | 8192 |
| halo | SSM / hybrid | any | 1024 | 4096 |
| halo | qwen4exp | any | 2048 | 4096 |
| standard | dense | any | 1024 | 2048 |
| standard | MoE | any | 2048 | 2048 |
| standard | SSM / hybrid | any | 1024 | 2048 |
| handheld | dense | any | 512 | 1024 |
| handheld | MoE | any | 1024 | 2048 |

The phase-1 scoring loop also tries alternative (batch, ubatch) pairs from
the candidates list (the optimistic default + 1024, 2048, 4096, 8192 in
various combinations) and picks the first that fits within the GPU budget
+ system memory check. Memory pressure is the actual decision driver;
the scoring is just a tiebreaker.

### Override precedence (low to high)

1. Built-in defaults (solver starts here)
2. System detection (GPU memory, hardware tier)
3. Solver output
4. User overrides via env vars and CLI flags

User overrides recognized by `apply_user_overrides()`:
- `LLAMA_CTX_SIZE` / `--ctx-size`
- `KV_CACHE_K_OVERRIDE` / `KV_CACHE_V_OVERRIDE`
- `LLAMA_THREADS_OVERRIDE` / `--threads`
- `MOE_UBATCH_OVERRIDE` / `OVERRIDE_UBATCH_SIZE` / `--ubatch-size`
- `OVERRIDE_CACHE_RAM` / `--cache-ram`
- `--no-ssd-cache`
- `--fit on`

### Legacy preset table (no longer used)

The `--noauto` escape hatch was deprecated in 2026-08; the solver is now
the only path. The historical preset table is kept here for reference
only — these values were the pre-solver defaults and may not match
current behavior. See [SOLVER.md](SOLVER.md) for the live tuning logic.

See [SOLVER.md](SOLVER.md) for the full algorithm, benchmark data, and edge-case
analysis.

---

## GPU memory budget

GPU-visible memory = VRAM carveout (`mem_info_vram_total`) + GTT
(`mem_info_gtt_total`). On UMA both are the same DRAM; GTT pages are mapped
from system RAM on demand.

### Nimo Axis N161 (Strix Halo, primary target)

- BIOS carves out **512 MiB** of VRAM (the documented 96 GB carveout is NOT
  active)
- OS sees ~125 GB RAM; GTT defaults to ~62.5 GB when `amdgpu.gttsize` is unset
- GPU-visible total: ~63 GB
- `detect-gpu.sh` recognizes Strix Halo via PCI ID (`1002:1586`) regardless of
  BIOS VRAM allocation
- Raise GTT: `sudo ./scripts/apply-ttm-kernel-params.sh 104` (~104 GiB ceiling)
- Practical ceiling is ~108 GB GTT (110 GB caused segfaults loading large models)

If the BIOS carveout is restored to 96 GB, the GTT kernel params are no longer
required (default ~16 GB GTT is sufficient), and `LLAMA_HARDWARE_TIER` switches
from `standard` to `halo` automatically.

### RADV APU 2/3 split

RADV on APUs (`has_dedicated_vram=false`) reports only 2/3 of (VRAM + GTT) as
the DEVICE_LOCAL heap and 1/3 as host heap (game-compat heuristic in
`radv_physical_device.c`). `~/.drirc` enables
`radv_enable_unified_heap_on_apu` for `llama-server`/`llama-cli`/`llama-bench`
so DEVICE_LOCAL = full VRAM + GTT. Without it, models > 2/3 of GPU-visible
memory crash with `vk::DeviceLostError` at load.

### Secondary: Ayaneo Flip KB

7840U / gfx1103 / Radeon 780M, 32 GB physical RAM, 6 GB VRAM carveout via
`amdgpu.vis_vramlimit=6144`, 18 GB GTT via `amdgpu.gttsize=18432`, ~26 GB
available to OS.

### Tertiary: Minisforum UM580 "zaphod"

5800H / gfx90c / 16 GB RAM, 24 GB GTT.

### Pre-flight guard

`llama-run.sh` checks if the model exceeds the GPU-visible budget (2 GiB
reserved for compute/staging). If it does, the server fails with a clear
message instead of the `radv/amdgpu: Not enough memory for command
submission` DeviceLost crash.

---

## Code style

All scripts are bash. Follow these conventions:

- `set -euo pipefail` at top
- `$(command)` for expansion (not backticks)
- `[[ ]]` for conditionals
- `function_name()` for functions
- 4-space indent
- `SCRIPT_DIR` / `PROJECT_ROOT` for paths (never hardcode)
- Use `log_info()`, `log_ok()`, `log_warn()`, `log_error()` for output

Logging helpers:
```bash
log_info()  { echo -e "\033[0;34m[INFO]\033[0m $1"; }
log_ok()    { echo -e "\033[0;32m[OK]\033[0m   $1"; }
log_warn()  { echo -e "\033[0;33m[WARN]\033[0m   $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
```

### Variable naming

| Pattern | Purpose | Examples |
|---------|---------|----------|
| `LLAMA_*` | User-facing defaults and overrides | `LLAMA_THREADS`, `LLAMA_CTX_SIZE` |
| `*_OVERRIDE` | User overrides that win over solver | `KV_CACHE_K_OVERRIDE`, `LLAMA_THREADS_OVERRIDE` |
| `SOLVER_*` | Solver output (internal) | `SOLVER_CTX_SIZE`, `SOLVER_K_TYPE` |
| `EXTRA_SERVER_ARGS` | Accumulated server arguments | (string, appended to throughout) |
| `_SSD_DISABLE` | Internal flags | (private, prefixed with `_`) |

### Sourcing pattern

`llama-run.sh` sources `detect-gpu.sh` and `optimize.sh` at the top:
```bash
source "$PROJECT_ROOT/scripts/detect-gpu.sh"
source "$PROJECT_ROOT/scripts/optimize.sh"
```

Shared functions in `detect-gpu.sh` and `optimize.sh` are available in
`llama-run.sh`'s scope. Do not duplicate functions across files.

---

## Testing

### Download integrity tests

Run after changing Hugging Face discovery, cache handling, or download
verification in `llama-run.sh`:

```bash
./tests/test-download-integrity.sh
```

Tests split-shard model download, cache validation, and integrity verification.

### Solver tests

```bash
# Unit test harness
./scratch/test-solver.sh

# Multi-model comparison
./scratch/test-solver-multi.sh

# pp/tg sweep across ubatch + KV types
./scratch/bench-solver-sweep.sh
```

### Benchmark tests

```bash
# Full benchmark with server + HTTP requests
./scripts/benchmark.sh --model Qwen3.6-35B-A3B --prompt-size 15000

# llama-bench sweeps
./src/cachy-llama-vulkan/build/bin/llama-bench -m model.gguf
```

### C++ tests

For CachyLLama C++ tests (test-backend-ops, test-sampling, etc.), see
[CachyLLama/AGENTS.md](CachyLLama/AGENTS.md#testing).

---

## Key constraints

- **Self-contained** — no system ROCm. The ROCm SDK is downloaded by
  `rebuild.sh` into `deps/`.
- **Gitignored** — `deps/`, `models/`, `kv-cache/`, `build/` directories are
  not in version control.
- **Submodule** — `CachyLLama/` is a git submodule. Clone with
  `--recurse-submodules`.
- **Vulkan default** — ROCm has stability issues on RDNA3 (GLM-4.7-Flash and
  DeepSeek2 MLA produce zero generation tokens on ROCm).
- **No patches** — `CachyLLama/` is maintained directly in git history. The
  old `patches/` directory is not used.
- **llama-cli blocks** — use `llama-server` (HTTP API) for testing model
  functionality instead of `llama-cli` interactive mode.
- **Never commit handoff files** — `ai-assisted/` is internal session context.
  Before every commit: `git reset HEAD ai-assisted/` if it appears staged.

---

## Common commands

```bash
# Build
./scripts/rebuild.sh

# Start server
./llama-run.sh --server

# Print solver profile
./llama-run.sh --print-profile Qwen3.6-35B-A3B

# List models
./llama-run.sh --list-models

# Download model
./llama-run.sh --download Qwen3-14B --quant Q4_K_M

# Raise GTT
sudo ./scripts/apply-ttm-kernel-params.sh 104

# Run tests
./tests/test-download-integrity.sh
./scratch/test-solver.sh

# Run benchmarks
./scripts/benchmark.sh --model Qwen3.6-35B-A3B --prompt-size 15000

# Format check (bash)
shellcheck scripts/*.sh llama-run.sh

# Search code
grep -rn "pattern" scripts/ llama-run.sh
```

---

## Session handoff

When ending a session, always create a handoff directory:

```
ai-assisted/YYYYMMDD/HHMM/
├── CONTINUATION_PROMPT.md  [MANDATORY] - Next session's complete context
├── AGENT_PLAN.md           [MANDATORY] - Remaining priorities & blockers
└── NOTES.md                [OPTIONAL]  - Technical notes
```

**NEVER commit handoff files.** Before every commit:
```bash
git status  # verify no ai-assisted/ staged
git reset HEAD ai-assisted/  # if it appears
git add -A && git commit -m "type(scope): description"
```

---

## License

**Source code:** [GPL-3.0-or-later](LICENSE)
**Documentation:** [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

CachyLLama (the inference engine in the submodule) is MIT-licensed (same as
upstream llama.cpp). See [CachyLLama/AGENTS.md](CachyLLama/AGENTS.md) for
C++ development conventions.
