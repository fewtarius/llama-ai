# AGENTS.md

Technical reference for AI agents working on this project.

## Directory structure

```
llama-ai/
├── llama-run.sh              # Main entry point — model detection, server launch
├── CachyLLama/               # Submodule — fork of ggml-org/llama.cpp
├── scripts/
│   ├── rebuild.sh            # Download ROCm SDK + build both backends
│   ├── env.sh                # Environment setup (source before ROCm tools)
│   ├── detect-gpu.sh         # GPU/APU auto-detection library
│   ├── benchmark.sh          # Performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
├── src/
│   ├── cachy-llama-rocm/     # ROCm build
│   │   ├── build.sh          # Build script (references $PROJECT_ROOT/CachyLLama)
│   │   └── build/            # Build output (binaries, libs)
│   └── cachy-llama-vulkan/   # Vulkan build
│       ├── build.sh          # Build script (references $PROJECT_ROOT/CachyLLama)
│       └── build/            # Build output (binaries, libs)
├── deps/                     # ROCm SDK (downloaded, gitignored)
├── models/                   # GGUF files (gitignored)
├── kv-cache/                 # Persistent KV cache (gitignored)
├── ssd-cache/                # SSD-backed KV cache (gitignored)
├── tests/
│   └── test-download-integrity.sh  # Download cache and split-shard validation
├── scratch/                  # Transient working files (gitignored)
└── patches/                  # DEPRECATED - kept for historical reference only
```

## Build

```bash
# Full setup from fresh checkout (builds Vulkan backend by default)
./scripts/rebuild.sh

# Full rebuild with ROCm support (adds --rocm flag)
./scripts/rebuild.sh --rocm

# Build Vulkan only (default)
./scripts/rebuild.sh

# Build ROCm only (optional - ROCm has stability issues on RDNA3)
./scripts/rebuild.sh --rocm

# Build both backends
./scripts/rebuild.sh --rocm  # (Vulkan is always built by default)
```

Build scripts in `src/cachy-llama-rocm/build.sh` and `src/cachy-llama-vulkan/build.sh` reference `$PROJECT_ROOT/CachyLLama` for the source.

`scripts/rebuild.sh` automatically applies patches from `patches/` to the submodule before building. Patches are checked for idempotency — if already applied, they're skipped.
Note: patch application is deprecated since we now maintain CachyLLama directly.
The `patches/` directory is kept for historical reference only.

## Tests

Run the model download integrity tests after changing Hugging Face discovery,
cache handling, or download verification in `llama-run.sh`:

```bash
./tests/test-download-integrity.sh
```

## Environment

```bash
# Required before using Vulkan tools (default)
source scripts/env.sh vulkan

# Or for ROCm (optional - ROCm has stability issues on RDNA3)
source scripts/env.sh rocm
```

Sets `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`, `PATH`.

## GPU Detection

`scripts/detect-gpu.sh` auto-detects the AMD GPU via PCI ID and sets:
- `HSA_OVERRIDE_GFX_VERSION` (e.g. `11.0.3` for Phoenix, `11.5.1` for Strix Halo)
- `LLAMA_GFX_ARCH` (e.g. `gfx1103`, `gfx1151`)
- `LLAMA_GPU_NAME` (e.g. `Radeon 780M`, `Radeon 8060S`)
- `LLAMA_THREADS` (optimal thread count)
- `LLAMA_TOTAL_RAM_GB` / `LLAMA_APU_VRAM_GB` / `LLAMA_RECOMMENDED_GTT_GB`
- `LLAMA_IS_STRIX_HALO` (`1` on Strix Halo, `0` elsewhere) — true/false
  signal that `llama-run.sh` reads to pick the halo preset. Detected via
  PCI ID `1002:1586` or `1002:1660` (exclusive to Strix Halo silicon), with
  a fallback of `gfx1150/gfx1151/gfx1152` plus `LLAMA_TOTAL_RAM_GB >= 64`.
- `LLAMA_HARDWARE_TIER` (`handheld` / `standard` / `halo`) - kept for
  back-compat, derived from `LLAMA_APU_VRAM_GB`. New code in `llama-run.sh`
  branches on `LLAMA_IS_STRIX_HALO` rather than the tier string.

User overrides via environment:
```bash
LLAMA_GFX_VERSION_OVERRIDE=11.0.3  # skip detection
LLAMA_GTT_SIZE=18                  # override GTT recommendation
LLAMA_IS_STRIX_HALO_OVERRIDE=1     # force halo preset (e.g. Apple Silicon with 64+ GB unified memory)
LLAMA_HARDWARE_TIER_OVERRIDE=halo  # mirror of the above for the tier string
```

Or via the CLI flag on `llama-run.sh` itself:
```bash
./llama-run.sh --model ... --hardware-tier halo    # or --hardware-tier standard
./llama-run.sh --model ... --is-strix-halo         # shorthand for halo
./llama-run.sh --model ... --no-strix-halo         # shorthand for standard
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
- `CachyLLama/` is a git submodule — use `--recurse-submodules` when cloning
- **Vulkan (RADV) is the default backend** — ROCm has stability issues on RDNA3
  (GLM-4.7-Flash/DeepSeek2 MLA produces zero generation tokens on ROCm)
- **Primary target: Nimo Axis N161 (Strix Halo "max")** — AMD Ryzen AI Max+ 395,
  Radeon 8060S iGPU on RDNA3.5 / gfx1151, 128GB unified memory. As of 2026-08 the
  machine boots with a **512 MiB BIOS VRAM carveout** (the documented 96GB
  carveout is NOT active — OS sees ~125GB RAM, GPU only 512 MiB VRAM). GTT is
  raised to 112 GiB via kernel params (`amdgpu.gttsize=114688 ttm.pages_limit=
  29360128 ttm.page_pool_size=29360128`, see GPU memory budget below), so
  GPU-visible ≈ 112.5 GiB. The 512 MiB carveout is the **supported state**
  on this hardware — `detect-gpu.sh` recognises Strix Halo via PCI ID
  (`1002:1586`) regardless of BIOS VRAM allocation. If the BIOS carveout is
  restored to 96GB the GTT kernel params are no longer required (default
  ~16GB GTT is sufficient when VRAM is large), and `LLAMA_HARDWARE_TIER`
  switches from `standard` to `halo` automatically.
- **RADV APU memory split** — RADV reports only 2/3 of (VRAM+GTT) as the
  DEVICE_LOCAL heap on APUs (game-compat heuristic in `radv_physical_device.c`:
  "report 2/3 as VRAM and 1/3 as GTT"). With the 512 MiB carveout that caps
  DEVICE_LOCAL at ~42 GiB. `~/.drirc` enables `radv_enable_unified_heap_on_apu`
  for `llama-server`/`llama-cli`/`llama-bench` so DEVICE_LOCAL = full
  VRAM+GTT. Without it, models > 2/3 of GPU-visible memory crash with
  `vk::DeviceLostError` at load; `llama-run.sh` now has a pre-flight guard that
  fails with a clear message instead.
- **Secondary: Ayaneo Flip KB** (7840U / gfx1103 / Radeon 780M, 32GB physical RAM,
  6GB VRAM carveout via `amdgpu.vis_vramlimit=6144`, 18GB GTT via
  `amdgpu.gttsize=18432`, ~26GB available to OS).
- **Tertiary: Minisforum UM580 "zaphod"** (5800H / gfx90c / 16GB RAM)

## Profile selection

`llama-run.sh` picks a preset from a small table based on three signals:
`LLAMA_IS_STRIX_HALO` (true/false), model kind (MoE / dense / SSM, read
from the GGUF header), and model size. Tier is no longer a separate axis
— the only meaningful boundary is Strix Halo vs everything else.

| Preset | When | ctx (default) | KV | batch | SSD cache | `--cpu-moe` |
|--------|------|---------------|-----|-------|-----------|-------------|
| `halo-moe-large` | Strix Halo, MoE, >50 GB | 131072 | q8_0 | 2048/512 | off | — |
| `halo-moe-small` | Strix Halo, MoE, ≤50 GB | 196608 | f16 | 2048/512 | off | — |
| `halo-dense` | Strix Halo, dense (any size) | 131072 | f16 | 2048/512 | off | — |
| `std-moe-large` | non-Halo, MoE, >18 GB | 65536 | q8_0 | 1024/256 | on | auto¹ |
| `std-moe-small` | non-Halo, MoE, ≤18 GB | 32768 | q8_0 | 1024/256 | on | auto¹ |
| `std-dense-large` | non-Halo, dense, >15 GB | 32768 | q4_0 | 1024/256 | on | — |
| `std-dense-small` | non-Halo, dense, ≤15 GB | 32768 | q8_0 | 1024/256 | on | — |
| `ssm` | Mamba / Jamba / Falcon-H1 / RWKV | 65536 (262144 on halo) | q8_0 | 1024/512 | off | — |

¹ `--cpu-moe + --load-mode none` is added automatically when the model is
≥23 GiB to avoid the mmap + `--moe-expert-residency` page-fault pathology
(see LTM). Halo tier never triggers this because GTT holds the model.

Override via `--hardware-tier halo|standard|handheld` (CLI flag) or
`LLAMA_HARDWARE_TIER_OVERRIDE` env var. Per-knob overrides still work:
`--checkpoint-min-step N`, `--ctx-checkpoints N`, `--cache-ram N`,
`--ubatch-size N`, `--ctx-size N`, `--no-ssd-cache`.

## GPU memory budget

GPU-visible memory = VRAM carveout (`mem_info_vram_total`) + GTT
(`mem_info_gtt_total`). On UMA both are the same DRAM; GTT pages are mapped
from system RAM on demand.

- Check current state:
  `cat /sys/class/drm/card0/device/mem_info_{vram,gtt}_total`
- Raise GTT: `sudo ./scripts/apply-ttm-kernel-params.sh 104` (GiB; writes
  `amdgpu.gttsize`, `ttm.pages_limit`, `ttm.page_pool_size` into the
  systemd-boot entry and requires a reboot). `amdgpu.gttsize` is deprecated on
  modern kernels but still honored; `ttm.pages_limit`/`ttm.page_pool_size` are
  the canonical way. On the AI Max+ 395, ~108GB GTT is the practical ceiling
  (110GB caused segfaults loading large models).
- SteamFork dual-ESP installs (Nimo): the real loader entries live on an
  UNMOUNTED ESP (`nvme1n1p1`); `/boot` is a redundant empty ESP.
  `apply-ttm-kernel-params.sh` now probes other vfat partitions and edits the
  entry in place. SteamFork OS updates regenerate the loader entry, so
  re-run the script after updates.
- RADV APU 2/3 split: DEVICE_LOCAL heap = 2/3 of (VRAM+GTT) unless
  `radv_enable_unified_heap_on_apu` is enabled for the app (~/.drirc, scoped to
  llama tools).
- `llama-run.sh` pre-flight: fails with a clear message when the model exceeds
  the GPU-visible budget (2 GiB reserved for compute/staging) instead of the
  `radv/amdgpu: Not enough memory for command submission` DeviceLost crash.

## Patch management

The `patches/` directory is **deprecated**. We now maintain `CachyLLama` directly as a fork rather than applying patches to an upstream submodule. The git history in `CachyLLama/` is the canonical source of truth.

### Historical reference

The old patch workflow (generating incremental and consolidated patches after each commit) is no longer needed. All custom changes are committed directly to the `CachyLLama/` fork's git history.

## License

Source code: GPL-3.0-or-later (see LICENSE)
Documentation: CC-BY-NC-SA-4.0
