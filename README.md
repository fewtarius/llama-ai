# llama-ai

Local LLM inference on AMD APU hardware using [llama.cpp](https://github.com/ggml-org/llama.cpp). Self-contained - no system ROCm install required.

## Quick start

```bash
git clone --recurse-submodules https://github.com/fewtarius/llama-ai.git
cd llama-ai

# Build both backends (downloads ROCm SDK, builds llama.cpp)
./scripts/rebuild.sh

# Drop a GGUF model in models/, then:
./llama-run.sh --server
# -> http://localhost:9090
```

To rebuild from scratch:

```bash
./scripts/rebuild.sh --rebuild
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

# Start server (auto-detects model)
./llama-run.sh --server

# Specific model and backend
./llama-run.sh --server gemma-4-26b --backend vulkan

# Download a model
./llama-run.sh --download Qwen3-14B --quant Q4_K_M

# List available backends
./llama-run.sh --list-backends

# Rebuild options
./scripts/rebuild.sh --rocm      # ROCm only
./scripts/rebuild.sh --vulkan    # Vulkan only
./scripts/rebuild.sh --rebuild   # Full rebuild from scratch
```

## How it works

Models are auto-profiled based on filename characteristics. MoE models get checkpoint strategies and reasoning format; SSM/Mamba models get context-shift disabled; large dense models get optimized batch sizes. Profiles are assigned dynamically - no hard-coded model names.

SSD-backed KV cache persists conversation state across server restarts. Enabled automatically for MoE models, or manually via `--cache-ssd PATH` flags.

## Benchmarking

```bash
./scripts/benchmark.sh
```

Runs 2-turn requests against each model to measure prompt caching performance. Tests both Vulkan and ROCm backends. Edit the `MODELS` array in the script to change what gets benchmarked.

## Benchmarks

100 tokens, ctx 32768, all GPU layers.

Ayaneo Flip KB (7840U / 780M / gfx1103 / 6GB VRAM + 18GB GTT):

| Model | Vulkan | ROCm |
|-------|--------|------|
| GLM-4.7-Flash Q4_K_M | 25.2 t/s | 9.4 t/s |
| Qwen3-14B Q5_K_M | 8.2 t/s | 6.2 t/s |
| Gemma 4 26B Q5_K_M | 18.0 t/s | 11.0 t/s |
| Qwen3.5-27B Q5_K_XL | 3.9 t/s | 2.7 t/s |
| Qwen3.6 35B Q4_K_XL | 20.6 t/s | 12.4 t/s |

Minisforum UM580 (5800H / Cezanne / gfx90c / 32GB RAM, 18GB GTT):

| Model | Vulkan |
|-------|--------|
| GLM-4.7-Flash Q4_K_M | 14.7 t/s |
| Qwen3-14B Q5_K_M | 4.2 t/s |
| Qwen3.5-27B Q5_K_XL | 2.1 t/s |

Gemma 26B and Qwen3.6 35B failed to load on 16GB (OOM). Vulkan is generally faster than ROCm on these APUs.

## Structure

```
├── llama-run.sh              # Main entry point
├── llama.cpp/                # Submodule - ggml-org/llama.cpp
├── scripts/
│   ├── rebuild.sh            # Download ROCm + build both backends
│   ├── env.sh                # Environment setup (source before using ROCm tools)
│   ├── detect-gpu.sh         # GPU/APU auto-detection library
│   ├── benchmark.sh          # Performance testing
│   └── apply-ttm-kernel-params.sh  # GPU memory config (GRUB + systemd-boot)
├── src/
│   ├── llama-cpp-rocm/       # ROCm build output + build.sh
│   └── llama-cpp-vulkan/     # Vulkan build output + build.sh
├── patches/                  # Patches applied to llama.cpp during build
├── deps/                     # ROCm SDK (downloaded by rebuild.sh)
├── models/                   # GGUF files
└── kv-cache/                 # Persistent KV cache
└── ssd-cache/                # SSD-backed KV cache (gitignored)
```

## License

Source code: [GPL-3.0-or-later](LICENSE)
Documentation: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

llama.cpp is MIT-licensed. ROCm components carry AMD's license.
