# AGENTS.md

Technical reference for AI agents working on this project.

## Directory structure

```
llama-ai/
├── llama-run.sh              # Main entry point — model detection, server launch
├── llama.cpp/                # Submodule — ggml-org/llama.cpp
├── scripts/
│   ├── rebuild.sh            # Download ROCm SDK + build both backends
│   ├── env.sh                # Environment setup (source before ROCm tools)
│   ├── detect-gpu.sh         # GPU/APU auto-detection library
│   ├── benchmark.sh          # Performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
├── src/
│   ├── llama-cpp-rocm/       # ROCm build
│   │   ├── build.sh          # Build script (references $PROJECT_ROOT/llama.cpp)
│   │   └── build/            # Build output (binaries, libs)
│   └── llama-cpp-vulkan/     # Vulkan build
│       ├── build.sh          # Build script (references $PROJECT_ROOT/llama.cpp)
│       └── build/            # Build output (binaries, libs)
├── deps/                     # ROCm SDK (downloaded, gitignored)
├── models/                   # GGUF files (gitignored)
├── kv-cache/                 # Persistent KV cache (gitignored)
├── ssd-cache/                # SSD-backed KV cache (gitignored)
├── scratch/                  # Transient working files (gitignored)
└── patches/                  # llama.cpp patches (gitignored, generated locally)
```

## Build

```bash
# Full setup from fresh checkout (downloads ROCm SDK + builds both backends)
./scripts/rebuild.sh

# Full rebuild (wipe deps/, re-download SDK, rebuild)
./scripts/rebuild.sh --rebuild

# Build individual backends
./scripts/rebuild.sh --rocm
./scripts/rebuild.sh --vulkan
```

Build scripts in `src/llama-cpp-rocm/build.sh` and `src/llama-cpp-vulkan/build.sh` reference `$PROJECT_ROOT/llama.cpp` for the source.

`scripts/rebuild.sh` automatically applies patches from `patches/` to the submodule before building. Patches are checked for idempotency — if already applied, they're skipped.

## Environment

```bash
# Required before using ROCm tools or running ROCm server
source scripts/env.sh rocm

# Or for Vulkan
source scripts/env.sh vulkan
```

Sets `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`, `PATH`.

## GPU Detection

`scripts/detect-gpu.sh` auto-detects the AMD GPU via PCI ID and sets:
- `HSA_OVERRIDE_GFX_VERSION` (e.g. `11.0.3` for Phoenix)
- `LLAMA_GFX_ARCH` (e.g. `gfx1103`)
- `LLAMA_GPU_NAME` (e.g. `Radeon 780M`)
- `LLAMA_THREADS` (optimal thread count)
- `LLAMA_TOTAL_RAM_GB` / `LLAMA_RECOMMENDED_GTT_GB`

User overrides via environment:
```bash
LLAMA_GFX_VERSION_OVERRIDE=11.0.3  # skip detection
LLAMA_GTT_SIZE=18                  # override GTT recommendation
```

The GPU map (`GPU_MAP` array in `detect-gpu.sh`) maps PCI device IDs to GFX versions. To add a new device, find your PCI ID with `lspci -nn | grep VGA` and add an entry.

## Code style

All scripts are bash:

- `set -euo pipefail` at top
- `$(command)` for expansion (not backticks)
- `[[ ]]` for conditionals
- `function_name()` for functions
- 4-space indent
- `SCRIPT_DIR` / `PROJECT_ROOT` for paths (never hardcode)

Logging helpers:
```bash
log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
log_ok()   { echo -e "\033[0;32m[OK]\033[0m   $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
```

## Key constraints

- Everything self-contained in this directory — no system ROCm
- `deps/`, `models/`, `kv-cache/`, `build/` directories are gitignored
- `llama.cpp/` is a git submodule — use `--recurse-submodules` when cloning
- Target hardware: Ayaneo Flip KB (7840U / 780M / gfx1103 / 32GB RAM), Minisforum UM580 (5800H / gfx90c / 16GB RAM)
- Vulkan (RADV) is generally faster than ROCm for inference on this GPU

## Patch management

The `patches/` directory contains custom patches applied to the `llama.cpp` submodule during builds. These patches are **gitignored** and kept locally for convenience - the canonical source of truth is the `llama.cpp` submodule fork itself, which is patched directly.

### Directory structure

```
patches/
├── 0001-*.patch              # Incremental patches (one per commit)
├── 0002-*.patch
├── ...
└── hybrid-model-full-fix.patch  # Consolidated patch (all changes in one file)
```

- **Incremental patches**: One patch per commit, generated with `git format-patch`. Preserve full commit history.
- **Consolidated patch**: A single diff of all changes from the upstream baseline. Used for quick application.

### Critical rules

1. **Regenerate patches after committing to llama.cpp** (for local backup only):

   ```bash
   cd llama.cpp
   rm -f ../patches/000*.patch ../patches/hybrid-model-full-fix.patch
   git format-patch -N --start-number=1 -o ../patches/ <upstream-base>..HEAD
   git diff <upstream-base>..HEAD > ../patches/hybrid-model-full-fix.patch
   ```

   Replace `<upstream-base>` with the upstream commit the patches are based on (currently `d05fe1d7d`).

### How rebuild.sh uses patches

`scripts/rebuild.sh` applies patches from `patches/` to the submodule before building. It checks idempotency - if a patch is already applied (via `git am --check`), it's skipped. This means patches must be generated from a clean baseline that matches the submodule's upstream HEAD.

## License

Source code: GPL-3.0-or-later (see LICENSE)
Documentation: CC-BY-NC-SA-4.0
