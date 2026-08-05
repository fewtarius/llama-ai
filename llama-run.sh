#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Llama.cpp Unified Runner — ROCm/HIP Optimized
# =============================================================================
# Auto-scans ./models for available GGUF files
# Supports Vulkan, ROCm/HIP, CPU backends

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Auto-detect GPU and system resources
source "$PROJECT_ROOT/scripts/detect-gpu.sh"
THREADS="${LLAMA_THREADS:-$(nproc)}"

# =============================================================================
# Build paths - backend specific
# =============================================================================

# Platform detection (macOS needs different memory budgeting than Linux)
if [[ "$(uname -s)" == "Darwin" ]]; then
    IS_DARWIN=true
    IS_DARWIN_ARM=false
    if [[ "$(uname -m)" == "arm64" ]]; then
        IS_DARWIN_ARM=true
    fi
else
    IS_DARWIN=false
    IS_DARWIN_ARM=false
fi

MODEL_DIR="$PROJECT_ROOT/models"

get_backend_binary() {
    local backend="$1"
    case "$backend" in
        rocm)
            echo "$PROJECT_ROOT/src/cachy-llama-rocm/build"
            ;;
        vulkan)
            echo "$PROJECT_ROOT/src/cachy-llama-vulkan/build"
            ;;
        metal)
            echo "$PROJECT_ROOT/src/cachy-llama-metal/build"
            ;;
        cpu)
            echo "$PROJECT_ROOT/src/cachy-llama-vulkan/build"
            ;;
        auto)
            # Check which is available
            if [[ -x "$PROJECT_ROOT/src/cachy-llama-metal/build/bin/llama-server" ]]; then
                echo "$PROJECT_ROOT/src/cachy-llama-metal/build"
            elif [[ -x "$PROJECT_ROOT/src/cachy-llama-rocm/build/bin/llama-server" ]]; then
                echo "$PROJECT_ROOT/src/cachy-llama-rocm/build"
            elif [[ -x "$PROJECT_ROOT/src/cachy-llama-vulkan/build/bin/llama-server" ]]; then
                echo "$PROJECT_ROOT/src/cachy-llama-vulkan/build"
            else
                echo ""
            fi
            ;;
        *)
            echo ""
            ;;
    esac
}

# =============================================================================
# Backend detection
# =============================================================================

detect_backend() {
    if [[ "$BACKEND" != "auto" ]]; then
        return 0
    fi

    # macOS: prefer Metal (only Apple Silicon has GPU acceleration)
    if [[ "$(uname -s)" == "Darwin" ]] && [[ -x "$PROJECT_ROOT/src/cachy-llama-metal/build/bin/llama-server" ]]; then
        BACKEND="metal"
        return 0
    fi

    # Check for Vulkan first (default backend - best stability on RDNA3)
    if [[ -x "$PROJECT_ROOT/src/cachy-llama-vulkan/build/bin/llama-server" ]]; then
        BACKEND="vulkan"
        return 0
    fi

    # Check for ROCm (optional backend - known issues with some archs)
    if [[ -x "$PROJECT_ROOT/src/cachy-llama-rocm/build/bin/llama-server" ]]; then
        BACKEND="rocm"
        return 0
    fi

    # Fallback
    if [[ "$(uname -s)" == "Darwin" ]]; then
        BACKEND="metal"
    else
        BACKEND="vulkan"
    fi
}

setup_backend_env() {
    if [[ -f "$PROJECT_ROOT/scripts/env.sh" ]]; then
        source "$PROJECT_ROOT/scripts/env.sh" "$BACKEND"
    fi
}

get_llama_binary() {
    local cmd="$1"  # server or cli
    local build_dir=$(get_backend_binary "$BACKEND")
    echo "$build_dir/bin/llama-$cmd"
}

# Returns total physical RAM in bytes (macOS / Linux compatible)
get_total_memory_bytes() {
    if [[ "$IS_DARWIN" == true ]]; then
        # hw.memsize returns bytes on macOS
        local bytes
        bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        echo "$bytes"
    else
        # /proc/meminfo on Linux
        awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

# Returns roughly available memory in bytes (free + inactive on macOS,
# MemAvailable on Linux). Conservative estimate.
get_available_memory_bytes() {
    if [[ "$IS_DARWIN" == true ]]; then
        local page_size free_pages inactive_pages
        page_size=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
        local vmstat_out
        vmstat_out=$(vm_stat 2>/dev/null)
        free_pages=$(echo "$vmstat_out" | awk "/Pages free/ {gsub(/\\./, \"\", \$3); print \$3}")
        inactive_pages=$(echo "$vmstat_out" | awk "/Pages inactive/ {gsub(/\\./, \"\", \$3); print \$3}")
        free_pages="${free_pages:-0}"
        inactive_pages="${inactive_pages:-0}"
        echo $(( (free_pages + inactive_pages) * page_size ))
    else
        awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

BACKEND="auto"
MODEL=""
MODEL_ALIAS=""
CTX_SIZE=65536
USER_CTX_SIZE=""  # set when user explicitly passes -c
N_PREDICT=256
GPU_LAYERS=99
KV_CACHE_TYPE_K="f16"
KV_CACHE_TYPE_V="f16"
USER_KV_CACHE_TYPE=""  # set when user explicitly passes --kv-cache-type
INTERACTIVE=false
PRINT_PROFILE=false
SERVER_MODE=false
PORT=9090
HOST="0.0.0.0"
EXTRA_COMMON_ARGS=""
EXTRA_SERVER_ARGS=""
OVERRIDE_REASONING=""
OVERRIDE_FIT=""
SSD_PATH=""
SSD_HOT_WINDOW="4096"
SSD_WARM_WINDOW=""
SSD_MAX_COLD="32"
SSD_PAGE_SIZE=""
SSD_HOT_RAM=""
SSD_WARM_RAM=""
PROMPT_MAX="8"
SSD_CHECKPOINTS="64"
# System prompt KV cache defaults (cross-conversation prompt sharing)
# Override with --cache-ssd-system-prompts / --cache-ssd-system-max-days
SSD_SYSTEM_PROMPTS="8"
SSD_NO_FSYNC=""
_SSD_DISABLE=false
SSD_SYSTEM_MAX_DAYS="30"
OVERRIDE_CHECKPOINT_EVERY=""
OVERRIDE_CTX_CHECKPOINTS=""
OVERRIDE_CACHE_RAM=""
OVERRIDE_REASONING_BUDGET=""
PRESERVE_REASONING="true"
OVERRIDE_N_PARALLEL=""
OVERRIDE_UBATCH_SIZE=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$1"; }
log_ok()    { printf '%b[OK]%b   %s\n' "$GREEN" "$NC" "$1"; }
log_warn()  { printf '%b[WARN]%b  %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1"; }

# Check whether the system is plugged into AC power. Returns 0 (true) when
# on AC, 1 (false) when on battery or when the state can't be determined.
#
# Linux: reads /sys/class/power_supply/AC*/online and /sys/class/power_supply/BAT*/status
# macOS: uses pmset -g batt; AC when "AC Power" appears in the output.
# Falls back to "AC assumed" on systems without either interface (desktops,
# servers) so the residency check doesn't block the standard tiers.
check_ac_power() {
    local on_ac=0
    if [[ -d /sys/class/power_supply ]]; then
        # Linux: any AC-type supply with online=1 means plugged in
        for ps in /sys/class/power_supply/AC*/online; do
            [[ -r "$ps" && "$(cat "$ps" 2>/dev/null)" == "1" ]] && on_ac=1
        done
        # Some handhelds (e.g. Ayaneo Flip KB) label AC as Mains or ADP.
        for ps in /sys/class/power_supply/{Mains,ADP}*/online; do
            [[ -r "$ps" && "$(cat "$ps" 2>/dev/null)" == "1" ]] && on_ac=1
        done
    fi
    if command -v pmset &>/dev/null && [[ $on_ac -eq 0 ]]; then
        # macOS: pmset shows "AC Power" line when plugged in
        pmset -g batt 2>/dev/null | grep -q "AC Power" && on_ac=1
    fi
    return $((1 - on_ac))
}

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat << USAGE
${BLUE}Llama.cpp Runner - Unified runner for Metal (macOS), Vulkan, ROCm, and CPU

${YELLOW}Usage:${NC}
    $0 [options] [model_or_file] [-- server options]
    $0 --download MODEL [--quant QUANT]

${YELLOW}Options:${NC}
    -b, --backend BACKEND    Backend: auto, rocm, vulkan, metal, cpu
    -m, --model MODEL       Model file or alias
    -a, --alias NAME        API model alias
    -t, --threads N         CPU threads (default: $THREADS)
    -c, --ctx-size N        Context size (default: $CTX_SIZE)
    -n, --n-predict N       Tokens to generate
    -ngl, --gpu-layers N    GPU layers (default: $GPU_LAYERS)
    --kv-cache-type TYPE    KV cache quantization (default: $KV_CACHE_TYPE_K)
    --interactive           Interactive chat mode
    --server                Run as API server
    --port PORT             Server port (default: $PORT)
    --fit                   Auto-fit GPU layers to available VRAM (disables -ngl)
    --host HOST             Server host (default: $HOST)
    --list-models           List available models
    --list-backends         List available backends
    --preserve-reasoning    Include reasoning/thinking in prior assistant messages
    --no-preserve-reasoning Strip reasoning from prior assistant messages (default)
    --reasoning-budget N    Max thinking tokens per response (default: 2048)
    --no-reasoning-budget   Disable thinking token limit
    --hardware-tier TIER    Override detected tier: halo, standard, or handheld
    --is-strix-halo         Force Strix Halo preset (shorthand for --hardware-tier halo)
    --no-strix-halo         Force non-Strix-Halo preset (shorthand for --hardware-tier standard)
    --no-ssd-cache          Disable SSD KV cache entirely (RAM prefix cache stays active)
    -h, --help              Show this help

${YELLOW}Download Model:${NC}
    --download MODEL        Download model from Hugging Face
    --quant QUANT          Quantization (default: Q4_K_M)
    --download-help        Show download help

${YELLOW}Examples:${NC}
    $0 --server gemma-4-26B
    $0 --backend vulkan --interactive Qwen3-14B
    $0 --server -m ./models/my-model.gguf --port 9091
    $0 --download Qwen3.6-35B
    $0 --download Qwen3.6-35B --quant Q5_K_M

USAGE
    exit 0
}

# =============================================================================
# Dynamic Profile Assignment
# =============================================================================
# Profiles are auto-detected from model characteristics, not hard-coded names
# This allows any model to work with optimized settings regardless of filename

# Global profile name for logging
profile_name=""

# Auto-scale --cache-ram to available memory. Models mmap the full
# file but only resident pages matter; on macOS unified memory, the OS will
# page out model pages under pressure. Still, we want a sane upper bound
# so the server doesn't fight the OS.
# Args: $1 = desired cache-ram in MiB (from profile), echoes adjusted value
adjust_cache_ram_for_memory() {
    local desired_mib="$1"
    local model_bytes="${MODEL_BYTES:-0}"
    local avail_bytes
    avail_bytes=$(get_available_memory_bytes)
    if [[ "$avail_bytes" -le 0 ]]; then
        echo "$desired_mib"
        return
    fi
    # Reserve: 4 GB for system overhead, plus resident model footprint.
    # When all layers are GPU-offloaded (-ngl >= 99), the model lives in VRAM/GTT
    # and doesn't consume system RAM. Only subtract model_resident for CPU models.
    local reserve_bytes=$((4 * 1024 * 1024 * 1024))
    local model_resident_bytes=0
    if [[ "$model_bytes" -gt 0 ]] && [[ "${GPU_LAYERS:-0}" -lt 99 ]]; then
        local total_bytes half_total
        total_bytes=$(get_total_memory_bytes)
        half_total=$((total_bytes / 2))
        if [[ "$model_bytes" -lt "$half_total" ]]; then
            model_resident_bytes=$model_bytes
        else
            model_resident_bytes=$half_total
        fi
    fi
    local max_cache_bytes=$((avail_bytes - reserve_bytes - model_resident_bytes))
    if [[ "$max_cache_bytes" -le 0 ]]; then
        echo 0
        return
    fi
    local max_cache_mib=$((max_cache_bytes / 1024 / 1024))
    if [[ "$desired_mib" -gt "$max_cache_mib" ]]; then
        echo "$max_cache_mib"
    else
        echo "$desired_mib"
    fi
}

# Scan the first 16KB of a GGUF file for architecture-defining keys.
# Updates is_moe and is_ssm (caller's locals) when the filename didn't
# already reveal the architecture (e.g. Qwen3-Coder-Next-UD uses neither
# "moe" nor "mamba" in its filename but is a MoE+SSM hybrid).
_scan_gguf_arch() {
    local gguf_path="$1"
    [[ ! -f "$gguf_path" || ! -r "$gguf_path" ]] && return 0
    local _tmp_header
    _tmp_header=$(mktemp /tmp/llama-scan-XXXXXX)
    dd if="$gguf_path" of="$_tmp_header" bs=16384 count=1 2>/dev/null || { rm -f "$_tmp_header"; return 0; }
    # MoE: any architecture with expert_count > 0
    if grep -q 'expert_count' "$_tmp_header" 2>/dev/null; then
        is_moe=true
    fi
    # Pure-SSM: has SSM layers but NO MoE experts and no attention layers.
    # Hybrid attention+SSM models (Qwen3.6-27B, GLM-4.7, etc.) have both
    # ssm.* and full_attention_interval < block_count — those need SSD
    # cache and context shifting, same as dense models.
    if grep -q 'ssm\.' "$_tmp_header" 2>/dev/null; then
        # Check for hybrid: if full_attention_interval is present, some
        # layers use full attention (hybrid), not pure SSM.
        if ! grep -q 'full_attention_interval' "$_tmp_header" 2>/dev/null; then
            if [[ "$is_moe" != true ]]; then
                is_ssm=true
            fi
        fi
    fi
    rm -f "$_tmp_header"
}
assign_profile() {
    local model_path="$1"
    local filename
    filename=$(basename "$model_path")
    local size_bytes
    # Detect split GGUF shards (e.g., model-00001-of-00003.gguf) and sum all shards.
    # stat on just the first shard (which may be only a few MB header) would
    # misclassify the model into the wrong profile tier.
    if [[ "$model_path" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        local shard_base="${model_path%-${BASH_REMATCH[1]}-of-${BASH_REMATCH[2]}.gguf}"
        local shard_count="${BASH_REMATCH[2]}"
        shard_count=$((10#$shard_count))
        size_bytes=0
        for ((i=1; i<=shard_count; i++)); do
            local shard_file
            shard_file=$(printf "%s-%05d-of-%05d.gguf" "$shard_base" "$i" "$shard_count")
            if [[ -f "$shard_file" ]]; then
                local sb
                sb=$(stat -c%s "$shard_file" 2>/dev/null || stat -f%z "$shard_file" 2>/dev/null || echo 0)
                size_bytes=$((size_bytes + sb))
            fi
        done
        [[ "$size_bytes" -eq 0 ]] && size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null || echo 0)
    else
        size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null || echo 0)
    fi
    MODEL_BYTES="$size_bytes"
    local size_gb=$((size_bytes / 1024 / 1024 / 1024))

    # Two-axis dispatch: is_strix_halo (PCI 1002:1586/1660) and model kind.
    # is_strix_halo is set by detect-gpu.sh; LLAMA_HARDWARE_TIER_OVERRIDE
    # can force it via env (e.g. for testing on Apple Silicon with >64GB).
    local is_strix_halo="false"
    [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]] && is_strix_halo="true"

    # Reset defaults. Profile presets override these as needed.
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
    KV_CACHE_TYPE_K="f16"
    KV_CACHE_TYPE_V="f16"
    GPU_LAYERS=99
    EXTRA_COMMON_ARGS=""
    EXTRA_SERVER_ARGS=""
    OVERRIDE_REASONING=""
    OVERRIDE_BATCH_SIZE=""
    OVERRIDE_N_PARALLEL="1"
    EXTRA_SERVER_ARGS+=" --no-mmproj"
    _SSD_DISABLE=false
    SSD_HOT_WINDOW=""
    SSD_WARM_WINDOW=""
    SSD_HOT_RAM=""
    SSD_WARM_RAM=""
    SSD_MAX_COLD=""

    # Detect model characteristics from filename. _scan_gguf_arch promotes
    # these for files whose names lack the keywords (e.g. Qwen3-Coder-Next-UD
    # is a MoE+SSM hybrid without "moe" or "mamba" in its filename).
    if echo "$filename" | grep -qiE "moe|a3b|a8b|flash|expert|gpt-oss"; then
        is_moe=true
    fi
    if echo "$filename" | grep -qiE "ssm|mamba|jamba|falcon-h1|rwkv"; then
        is_ssm=true
    fi

    _scan_gguf_arch "$model_path"

    # =========================================================================
    # GPU budget detection (VRAM + GTT - 2GiB reserve for compute/staging)
    # Used for MoE streaming decision: if model > GPU budget, pin experts to CPU.
    # =========================================================================
    _gpu_budget_bytes=0
    _gpu_budget_gb=0
    if [[ "$BACKEND" == "vulkan" || "$BACKEND" == "rocm" || "$BACKEND" == "metal" ]]; then
        for _c in /sys/class/drm/card[0-9]/device; do
            [[ -d "$_c" ]] || continue
            _vram=$(cat "$_c/mem_info_vram_total" 2>/dev/null || echo 0)
            _gtt=$(cat "$_c/mem_info_gtt_total" 2>/dev/null || echo 0)
            _gpu_budget_bytes=$(( _vram + _gtt - 2 * 1024 * 1024 * 1024 ))  # 2 GiB reserve
            _gpu_budget_gb=$(( _gpu_budget_bytes / 1073741824 ))
            break
        done
        # On macOS with Metal, unified memory - use total system RAM minus reserve
        if [[ "$IS_DARWIN" == "true" ]] && [[ "$_gpu_budget_bytes" -eq 0 ]]; then
            _total_mem=$(get_total_memory_bytes)
            _gpu_budget_bytes=$(( _total_mem - 4 * 1024 * 1024 * 1024 ))
            _gpu_budget_gb=$(( _gpu_budget_bytes / 1073741824 ))
        fi
    fi

    # MoE streaming decision: if MoE model exceeds GPU budget, use --cpu-moe
    # to pin expert weights to host RAM while keeping attention/embedding on GPU.
    # For non-MoE models, --cpu-moe is never used (dense models must fit entirely).
    _need_cpu_moe=false
    _need_moe_residency=false
    if [[ "$is_moe" == true ]] && [[ ${_gpu_budget_bytes:-0} -gt 0 ]]; then
        if [[ ${MODEL_BYTES:-0} -gt ${_gpu_budget_bytes} ]]; then
            _need_cpu_moe=true
            log_info "MoE expert weights pinned to CPU RAM (model ${size_gb}GB exceeds GPU budget ${_gpu_budget_gb}GB)"
        elif [[ ${MODEL_BYTES:-0} -gt $((_gpu_budget_bytes * 95 / 100)) ]]; then
            # Model fits but is tight (>95% of budget) - residency helps keep hot experts on GPU
            # when there's not enough headroom for the OS/Vulkan to manage all pages.
            _need_moe_residency=true
        fi
    fi

    # Tell _apply_moe_residency whether we need it (set by GPU budget check above)
    # Must be BEFORE preset calls _apply_moe_residency
    if [[ "$_need_moe_residency" == "true" ]]; then
        EXTRA_SERVER_ARGS+=" __NEEDS_MOE_RESIDENCY__"
    fi

    # Pick the preset function. Branches are flat - one selection per row.
    is_moe="${is_moe:-false}"
    is_ssm="${is_ssm:-false}"
    local preset
    if [[ "$is_ssm" == true ]]; then
        preset="_preset_ssm"
    elif [[ "$is_strix_halo" == "true" && "$is_moe" == true ]]; then
        if [[ $size_gb -gt 50 ]]; then preset="_preset_halo_moe_large"
        else preset="_preset_halo_moe_small"; fi
    elif [[ "$is_strix_halo" == "true" ]]; then
        preset="_preset_halo_dense"
    elif [[ "$is_moe" == true ]]; then
        if [[ $size_gb -gt 18 ]]; then preset="_preset_std_moe_large"
        else preset="_preset_std_moe_small"; fi
    elif [[ $size_gb -gt 15 ]]; then
        preset="_preset_std_dense_large"
    else
        preset="_preset_std_dense_small"
    fi

    # Apply preset (sets CTX_SIZE, KV types, batch, checkpoints, _SSD_DISABLE,
    # SSD window settings, profile_name, reasoning flags).
    $preset

    # DeepSeek-specific kernel/ubatch tuning (halo tier only).
    # DeepSeek MLA compresses KV via latent vectors; llama.cpp's MLA flash-attn
    # path is the fastest available kernel. 4096/4096 batch/ubatch amortizes
    # kernel launch overhead across the long reasoning prefills the model
    # produces. Halo has 96+ GB GTT so per-batch memory cost is not a
    # constraint. Standard tier (Flip, Cezanne) is untouched because 4096
    # ubatch OOMs the smaller iGPUs.
    # UNTESTED — applied on observation; verify on next server restart.
    _apply_deepseek_mla_tuning

    # Apply MoE flags based on GPU budget measurement
    if [[ "$_need_cpu_moe" == "true" ]]; then
        # --load-mode none (full read into RAM) only if model fits in system RAM.
        # If model > system RAM, keep mmap (default) so only needed pages are loaded.
        _total_mem=$(get_total_memory_bytes)
        if [[ ${MODEL_BYTES:-0} -lt $_total_mem ]]; then
            EXTRA_SERVER_ARGS+=" --cpu-moe --load-mode none"
        else
            EXTRA_SERVER_ARGS+=" --cpu-moe"
            log_info "Model (${size_gb}GB) exceeds system RAM (${_total_mem}GB) - keeping mmap, experts pinned to CPU"
        fi
    fi

    # Clamp --cache-ram to available memory on tight systems (e.g. 24 GB
    # macOS running 20+ GB MoE models). On Halo with 64+ GB RAM the headroom
    # check passes through and the value stays at the preset-set level.
    if [[ "${LLAMA_ADJUST_CACHE:-1}" == "1" ]]; then
        local _orig_cache_ram _new_cache_ram
        _orig_cache_ram=$(echo "$EXTRA_SERVER_ARGS" | sed -nE 's/.*--cache-ram ([0-9]+).*/\1/p')
        if [[ -n "$_orig_cache_ram" ]]; then
            _new_cache_ram=$(adjust_cache_ram_for_memory "$_orig_cache_ram")
            if [[ "$_new_cache_ram" -le 0 ]]; then
                EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed -E 's/ --cache-ram [0-9]+//')
                log_info "cache-ram disabled (insufficient memory headroom); SSD cache remains"
            elif [[ "$_new_cache_ram" -lt "$_orig_cache_ram" ]]; then
                EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed -E "s/--cache-ram [0-9]+/--cache-ram $_new_cache_ram/")
                log_info "cache-ram reduced: ${_orig_cache_ram} MiB -> ${_new_cache_ram} MiB (memory-constrained)"
            fi
        fi
    fi

    printf '%bAuto profile: %b%s%b (%sGB, MoE=%s, SSM=%s)%b\n' "$CYAN" "$GREEN" "$profile_name" "$NC" "$size_gb" "$is_moe" "$is_ssm" "$NC"
}

# =============================================================================
# Profile presets
# =============================================================================
# Each preset is a small bash function called by assign_profile() to set
# CTX_SIZE, KV cache types, batch/ubatch, checkpoint args, SSD-related
# flags, and the reasoning defaults for non-SSM models. Helpers below
# (_apply_reasoning_defaults etc.) factor out the common arg fragments.
#
# Naming: preset_<tier>_<kind>[_<size>]
#   tier: halo | std (everything else, including Strix Point and Apple Silicon)
#   kind: moe | dense | ssm
#   size: large | small (threshold depends on kind)
#
# Why two tiers? Halo is a one-of-a-kind machine: 64-128 GB unified memory,
# 40 RDNA3.5 CUs, GTT can be expanded to 100+ GB. SSD writes are pure
# overhead on the prefill critical path. Every other machine benefits from
# SSD-backed cross-restart warmup because the working set does not fit.

# Reasoning defaults for non-SSM models. Sets the sampling chain Unsloth
# recommends for Qwen 3.6 thinking mode and turns reasoning on with a
# 2048-token budget cap (prevents think loops in agentic workloads).
# https://unsloth.ai/docs/models/qwen3.6
_apply_reasoning_defaults() {
    EXTRA_SERVER_ARGS+=" --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00"
    EXTRA_SERVER_ARGS+=" --repeat-penalty 1.0 --presence-penalty 0.0"
    EXTRA_SERVER_ARGS+=" --reasoning-format auto"
    OVERRIDE_REASONING="on"
    OVERRIDE_REASONING_BUDGET="2048"
}

# SSD cache window defaults for non-Halo presets. Hot=4096, warm=8192,
# 960 MiB hot RAM, 1440 MiB warm RAM, 32 cold checkpoints.
_apply_ssd_defaults() {
    SSD_HOT_WINDOW="4096"
    SSD_WARM_WINDOW="8192"
    SSD_HOT_RAM="960"
    SSD_WARM_RAM="1440"
    SSD_MAX_COLD="32"
}

# --moe-expert-residency gate. On Halo we always enable - GTT holds hot
# experts cheaply. On standard tier we require AC power to enable (heavy
# SSD I/O from paging drains battery on handhelds). CPU-mode MoE
# (--cpu-moe + --load-mode none) actually skips residency entirely; this
# helper just adds the flag if residency is fine.
_apply_moe_residency() {
    local tier="$1"
    # Check for explicit marker set by assign_profile (not just --cpu-moe,
    # because we want residency for large-but-fitting models too)
    if [[ "$EXTRA_SERVER_ARGS" != *"__NEEDS_MOE_RESIDENCY__"* ]]; then
        return 0
    fi
    # Strip the marker
    EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed 's/ __NEEDS_MOE_RESIDENCY__//')
    if [[ "$tier" == "halo" ]]; then
        EXTRA_SERVER_ARGS+=" --moe-expert-residency"
    elif check_ac_power; then
        EXTRA_SERVER_ARGS+=" --moe-expert-residency"
    else
        log_warn "MoE expert residency skipped: standard tier on battery"
        log_warn "  SSD I/O during eviction drains battery and adds latency"
        log_warn "  Plug in AC power and re-run to enable --moe-expert-residency"
    fi
}

# DeepSeek MLA tuning. Bumps batch/ubatch to 4096 and adds --flash-attn.
# Halo-only by design - 4096 ubatch OOMs smaller iGPUs (Flip, Cezanne).
# Detection is filename-based (case-insensitive match on "deepseek").
_apply_deepseek_mla_tuning() {
    if [[ "$is_strix_halo" == "true" && "$is_moe" == "true" ]] \
       && echo "$filename" | grep -qiE "deepseek"; then
        OVERRIDE_BATCH_SIZE="--batch-size 4096 --ubatch-size 4096"
        EXTRA_SERVER_ARGS+=" --flash-attn auto"
        log_info "DeepSeek MLA tuning: batch/ubatch 4096/4096 + --flash-attn"
    fi
}

# -----------------------------------------------------------------------------
# Halo presets (Radeon 8060S / Ryzen AI Max+ 395, 64-128 GB unified memory)
# -----------------------------------------------------------------------------

_preset_halo_moe_large() {
    # >50 GB MoE: q8 KV (f16 wouldn't fit alongside cache-ram at this size),
    # 32K checkpoint step (8 checkpoints -> 32768 tokens), 2 slots = ~8 GiB
    # ring for these wide models (Qwen3-Coder-Next 80 GB, MiniMax M2.7 70 GB).
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=131072
    KV_CACHE_TYPE_K="f16"
    KV_CACHE_TYPE_V="f16"
    OVERRIDE_BATCH_SIZE="--batch-size 2048 --ubatch-size 512"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 32768 --ctx-checkpoints 2 --cache-ram 8192"
    _SSD_DISABLE=true
    _apply_reasoning_defaults
    _apply_moe_residency halo
    profile_name="halo-moe-large"
}

_preset_halo_moe_small() {
    # <=50 GB MoE: f16 KV (fits comfortably alongside cache-ram), 196K ctx.
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=196608
    KV_CACHE_TYPE_K="f16"
    KV_CACHE_TYPE_V="f16"
    OVERRIDE_BATCH_SIZE="--batch-size 2048 --ubatch-size 512"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 32768 --ctx-checkpoints 2 --cache-ram 8192"
    _SSD_DISABLE=true
    _apply_reasoning_defaults
    _apply_moe_residency halo
    profile_name="halo-moe-small"
}

_preset_halo_dense() {
    # Dense (any size): f16 KV, 131K ctx, 16 GB cache-ram, slot similarity
    # 0.15 for in-memory checkpoint reuse. SSD off (GTT holds everything).
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=131072
    KV_CACHE_TYPE_K="f16"
    KV_CACHE_TYPE_V="f16"
    OVERRIDE_BATCH_SIZE="--batch-size 2048 --ubatch-size 512"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 8192 --ctx-checkpoints 8 --cache-ram 16384 --no-checkpoint-near-end --slot-prompt-similarity 0.15"
    _SSD_DISABLE=true
    _apply_reasoning_defaults
    profile_name="halo-dense"
}

# -----------------------------------------------------------------------------
# Standard presets (everything else: Phoenix, Cezanne, Strix Point, dGPUs)
# -----------------------------------------------------------------------------

_preset_std_moe_large() {
    # >18 GB MoE: q8 KV, ubatch 256 (ubatch 512 GPU hard-locks small iGPUs
    # at ~3K tokens), 65K ctx default. Big models on this tier get --cpu-moe
    # + --load-mode none from assign_profile() (mmap+residency disaster).
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=65536
    KV_CACHE_TYPE_K="q8_0"
    KV_CACHE_TYPE_V="q8_0"
    OVERRIDE_BATCH_SIZE="--batch-size 1024 --ubatch-size 256"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 8192 --ctx-checkpoints 8 --cache-ram 4096 --no-checkpoint-near-end"
    _apply_ssd_defaults
    _apply_reasoning_defaults
    _apply_moe_residency standard
    profile_name="std-moe-large"
}

_preset_std_moe_small() {
    # <=18 GB MoE: same as large but smaller ctx, 4 checkpoints.
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
    KV_CACHE_TYPE_K="q8_0"
    KV_CACHE_TYPE_V="q8_0"
    OVERRIDE_BATCH_SIZE="--batch-size 1024 --ubatch-size 256"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 8192 --ctx-checkpoints 4 --cache-ram 4096 --no-checkpoint-near-end"
    _apply_ssd_defaults
    _apply_reasoning_defaults
    _apply_moe_residency standard
    profile_name="std-moe-small"
}

_preset_std_dense_large() {
    # >15 GB dense: q4 KV (compression matters on tight VRAM), 65K ctx.
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
    KV_CACHE_TYPE_K="q4_0"
    KV_CACHE_TYPE_V="q4_0"
    OVERRIDE_BATCH_SIZE="--batch-size 1024 --ubatch-size 256"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 8192 --ctx-checkpoints 4 --cache-ram 4096 --no-checkpoint-near-end --slot-prompt-similarity 0.15"
    _apply_ssd_defaults
    _apply_reasoning_defaults
    profile_name="std-dense-large"
}

_preset_std_dense_small() {
    # <=15 GB dense: q8 KV (no need to over-compress), smaller ctx.
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
    KV_CACHE_TYPE_K="q8_0"
    KV_CACHE_TYPE_V="q8_0"
    OVERRIDE_BATCH_SIZE="--batch-size 1024 --ubatch-size 256"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step 8192 --ctx-checkpoints 4 --cache-ram 4096 --no-checkpoint-near-end --slot-prompt-similarity 0.15"
    _apply_ssd_defaults
    _apply_reasoning_defaults
    profile_name="std-dense"
}

# -----------------------------------------------------------------------------
# SSM preset (Mamba / Jamba / Falcon-H1 / RWKV)
# -----------------------------------------------------------------------------
# SSM hidden state is constant-size; attention KV is a small fraction of
# total context. --cache-ssd path requires KV serialization llama_state_seq_*
# which doesn't apply to Mamba layers, so SSD is unconditionally off.

_preset_ssm() {
    if [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]]; then
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=262144
        EXTRA_SERVER_ARGS+=" --no-context-shift --ctx-checkpoints 0 --cache-ram 16384"
    else
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=65536
        # Floor 1 GB so sub-2 GB VRAM systems (Cezanne 512 MiB carveout)
        # don't produce a negative cache-ram.
        local _ssm_cache_ram=$(( LLAMA_APU_VRAM_GB * 1024 - 2048 ))
        (( _ssm_cache_ram < 1024 )) && _ssm_cache_ram=1024
        EXTRA_SERVER_ARGS+=" --no-context-shift --ctx-checkpoints 0 --cache-ram ${_ssm_cache_ram}"
    fi
    KV_CACHE_TYPE_K="q8_0"
    KV_CACHE_TYPE_V="q8_0"
    OVERRIDE_BATCH_SIZE="--batch-size 1024 --ubatch-size 512"
    OVERRIDE_REASONING="on"
    OVERRIDE_REASONING_BUDGET="2048"
    _SSD_DISABLE=true
    profile_name="ssm"
}


# =============================================================================
# Auto-discover models from ./models directory
# =============================================================================

# Lightweight model registry. Use parallel MODELS_NAME[] / MODELS_PATH[] arrays
# (macOS ships bash 3.2 which lacks associative arrays).
declare -a MODELS_NAME=()
declare -a MODELS_PATH=()

scan_models() {
    MODELS_NAME=()
    MODELS_PATH=()
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo -e "${YELLOW}Warning: Models directory not found: $MODEL_DIR${NC}"
        return
    fi

    # Scan for .gguf files in top-level of models dir
    while IFS= read -r -d '' file; do
        local basename
        basename=$(basename "$file" .gguf)
        MODELS_NAME+=("$basename")
        MODELS_PATH+=("$file")
    done < <(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" -print0 2>/dev/null)
}

# Initialize models from directory
scan_models

# =============================================================================
# Functions
# =============================================================================

list_models() {
    echo -e "${BLUE}Available Models:${NC}"
    echo -e "${BLUE}(auto-scanned from $MODEL_DIR)${NC}"
    echo ""
    
    local found=0
    # Sort output for consistent ordering
    local i
    for i in $(printf '%s\n' "${!MODELS_NAME[@]}" | sort -n); do
        local name="${MODELS_NAME[$i]}"
        local model="${MODELS_PATH[$i]}"
        if [[ -f "$model" ]]; then
            local size
            size=$(du -h "$model" 2>/dev/null | cut -f1)
            echo -e "  ${GREEN}$name${NC}  - $size"
            found=1
        fi
    done
    
    if [[ $found -eq 0 ]]; then
        echo -e "  ${YELLOW}No models found in $MODEL_DIR${NC}"
        echo -e "  ${CYAN}Place .gguf files in that directory${NC}"
    fi
    exit 0
}

list_backends() {
    # Single source of truth: list_backends_v2 covers rocm/vulkan/metal/cpu.
    list_backends_v2
}

list_backends_v2() {
    echo -e "${BLUE}Available Backends:${NC}"
    local binary_rocm="$PROJECT_ROOT/src/cachy-llama-rocm/build/bin/llama-cli"
    local binary_vulkan="$PROJECT_ROOT/src/cachy-llama-vulkan/build/bin/llama-cli"
    local binary_metal="$PROJECT_ROOT/src/cachy-llama-metal/build/bin/llama-cli"
    if [[ -x "$binary_rocm" ]]; then
        if [[ -n "$LLAMA_GPU_NAME" ]]; then
            echo -e "  ${GREEN}[*] ROCm/HIP${NC}   - $LLAMA_GPU_NAME ($LLAMA_GFX_ARCH)"
        else
            echo -e "  ${CYAN}[ ] ROCm/HIP${NC}   - installed (GPU not in detection map)"
        fi
    else
        echo -e "  ${YELLOW}[ ] ROCm/HIP${NC}   - not built"
    fi
    if [[ -x "$binary_vulkan" ]]; then
        echo -e "  ${GREEN}[*] Vulkan${NC}      - available"
    else
        echo -e "  ${YELLOW}[ ] Vulkan${NC}      - not built"
    fi
    if [[ -x "$binary_metal" ]]; then
        if [[ -n "$LLAMA_GPU_NAME" ]]; then
            echo -e "  ${GREEN}[*] Metal${NC}       - $LLAMA_GPU_NAME"
        else
            echo -e "  ${GREEN}[*] Metal${NC}       - available (Apple Silicon)"
        fi
    else
        echo -e "  ${YELLOW}[ ] Metal${NC}       - not built (macOS only)"
    fi
    echo -e "  ${GREEN}[*] CPU${NC}         - always available"
    exit 0
}

setup_performance() {
    # CPU frequency (needs sudo, graceful fallback)
    if command -v cpupower &>/dev/null; then
        sudo cpupower frequency-set -g performance 2>/dev/null || true
    fi
    
    # CPU energy performance preference - use performance governor during inference
    # balance_performance is too conservative for GPU-bound workloads where
    # the CPU handles graph construction and scheduling
    for p in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do
        echo performance | sudo tee "$p" >/dev/null 2>&1 || true
    done
    # Set GPU to auto (high causes near instant APU hangs)
    for card in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
        if [[ -f "$card" ]]; then
            echo "auto" | sudo tee "$card" >/dev/null 2>&1 || true
        fi
    done
}

# Restore power settings to balanced/auto after inference
restore_performance() {
    for p in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do
        echo balance_performance | sudo tee "$p" >/dev/null 2>&1 || true
    done
}

_cleanup() {
    restore_performance
}
trap _cleanup EXIT

setup_rocm_env() {
    export ROCM_PATH="$PROJECT_ROOT/deps"
    export HIP_PATH="$ROCM_PATH"
    export HIP_PLATFORM=amd
    export HIP_VISIBLE_DEVICES=0
    export HSA_OVERRIDE_GFX_VERSION="${LLAMA_GFX_VERSION:-11.0.3}"
    export LD_LIBRARY_PATH="${PROJECT_ROOT:-.}/deps/lib:${LD_LIBRARY_PATH:-}"
}

setup_vulkan_env() {
    export LD_LIBRARY_PATH="${PROJECT_ROOT:-.}/deps/lib:${LD_LIBRARY_PATH:-}"
    # Vulkan shader pipeline cache - larger cache reduces recompilation stalls
    export MESA_SHADER_CACHE_MAX_SIZE="${MESA_SHADER_CACHE_MAX_SIZE:-2G}"
    # Ensure cache directory exists and is persisted
    export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-$HOME/.cache/mesa_shader_cache}"
    mkdir -p "$MESA_SHADER_CACHE_DIR"
    # RADV performance tuning for compute workloads
    export RADV_PERFTEST="${RADV_PERFTEST:-gplp}"
    # CachyLLAMA RDNA3 (gfx1103, 7840U): measured +4.5% tg64 at nps=64 vs the
    # conservative nps=8 default. nps=100 gets another +0.1% but triples the
    # lockup_timeout risk on slower APUs. 64 is the sweet spot. See RDNA3_NOTES.md.
    # Only auto-sets on Phoenix; other RDNA3 silicon keeps the nps=8 default.
    # Cezanne (gfx90c) is intentionally NOT overridden: 2026-07-20 sweep on
    # zaphod (Minisforum UM580+, 8 Vega CUs) showed NPS sweep 1-100 gives
    # <3% tg variation across 7b/14b/27b/35B-A3B - within noise. The 7840U's
    # 12 RDNA3 CUs benefit from deeper submit queues; 8 Vega CUs don't.
    # See CachyLLama/CEZANNE_NOTES.md.
    # Manual override: export GGML_VK_NODES_PER_SUBMIT=N before invoking.
    if [[ "${LLAMA_GFX_ARCH:-}" == "gfx1103" ]] && [[ -z "${GGML_VK_NODES_PER_SUBMIT:-}" ]]; then
        export GGML_VK_NODES_PER_SUBMIT=64
    fi
}

setup_metal_env() {
    # Metal uses no special runtime env; unified memory is automatic on Apple Silicon.
    # Setting GPU_HONEST_CELLS=1 can help on devices with binned/inactive GPU cores
    # but is harmless on healthy hardware.
    export GGML_METAL_DEVICE_DEBUG=0
}

apply_backend_env() {
    case "$BACKEND" in
        rocm)   setup_rocm_env ;;
        vulkan) setup_vulkan_env ;;
        metal)  setup_metal_env ;;
    esac
}

# =============================================================================
# Model Download
# =============================================================================

# Query HuggingFace API for model information
# Returns JSON with model metadata
hf_api_get() {
    local endpoint="$1"
    local url="https://huggingface.co/api/$endpoint"
    curl -s -L --fail "$url" 2>/dev/null
}

# Search for models on HuggingFace matching a pattern
# Output: repo_id lines (owner/repo format)
hf_search_models() {
    local query="$1"
    local limit="${2:-10}"
    local encoded_query="${query// /+}"
    hf_api_get "models?search=$encoded_query&sort=downloads&direction=-1&limit=$limit" 2>/dev/null | \
        jq -r '.[].id // empty' 2>/dev/null
}

# List files in a HuggingFace repository
# Output: filenames
hf_list_files() {
    local repo="$1"
    hf_api_get "models/$repo" 2>/dev/null | \
        jq -r '.siblings[].rfilename // empty' 2>/dev/null
}

# Extract metadata for a specific file from Hugging Face model metadata on stdin.
hf_file_info() {
    local filename="$1"
    jq -c --arg filename "$filename" \
        'first(.siblings[]? | select(.rfilename == $filename)) // empty' 2>/dev/null
}

# Find GGUF files in a repo matching a quantization pattern
# Args: repo quant_pattern
# Output: matching filename(s), best match first
hf_find_gguf() {
    local repo="$1"
    local quant_pattern="${2:-Q4_K_M}"
    local files
    files=$(hf_list_files "$repo") || return 1
    
    # Normalize quant pattern for matching
    # Handle formats like "Q4_K_M", "UD-Q4_K_M", "IQ4_NL", etc.
    local quant_base="${quant_pattern#*-}"  # Remove prefix like "UD-"
    local quant_family="${quant_base%%_*}"  # Get base like "Q4"

    # Lowercase via tr for bash 3.2 compatibility (macOS default)
    local quant_lc quant_base_lc quant_family_lc
    quant_lc=$(printf '%s' "$quant_pattern" | tr 'A-Z' 'a-z')
    quant_base_lc=$(printf '%s' "$quant_base" | tr 'A-Z' 'a-z')
    quant_family_lc=$(printf '%s' "$quant_family" | tr 'A-Z' 'a-z')

    # Extract matching GGUF files
    local matches=()
    while IFS= read -r f; do
        [[ "$f" == *.gguf ]] || continue
        # Lowercase the filename for case-insensitive comparison
        local f_lc
        f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
        # Match: Q4_K_M style OR unsloth UD-Q4_K_M style OR IQ4_NL style
        # Order matters - most specific first
        if [[ "$f_lc" =~ [-_]${quant_lc}[-_.] ]] || \
           [[ "$f_lc" =~ [-_]${quant_lc}$ ]] || \
           [[ "$f_lc" =~ [-_]${quant_base_lc}[-_.] ]] || \
           [[ "$f_lc" =~ [-_]${quant_base_lc}$ ]] || \
           [[ "$f_lc" =~ [-_]${quant_family_lc}[-_] ]]; then
            matches+=("$f")
        fi
    done <<< "$files"
    
    # If no specific quant matches, return all GGUF files
    if [[ ${#matches[@]} -eq 0 ]]; then
        while IFS= read -r f; do
            [[ "$f" == *.gguf ]] && matches+=("$f")
        done <<< "$files"
    fi
    
    # Sort by size (prefer larger quants)
    printf '%s\n' "${matches[@]}" | sort -t'_' -k2 -V | tac
}

# Detect available quantization options in a repo
hf_list_quants() {
    local repo="$1"
    local files
    files=$(hf_list_files "$repo") 2>/dev/null || return 1
    
    # Extract unique quant types from filenames
    echo "$files" | grep -oE '[Qq][0-9]+[_-]?K?_?[SMXL]?' | sort -u | head -20
}

download_model() {
    local input="$1"
    local quant="${2:-Q4_K_M}"
    local target_dir="$MODEL_DIR"
    
    local repo=""
    local filename=""
    
    # Detect input format
    if [[ "$input" == *"/"* ]]; then
        # Direct repo format: owner/repo or owner/repo:quant
        repo="${input%%:*}"
        if [[ "$input" == *":"* ]]; then
            quant="${input##*:}"
        fi
    else
        # Search for model by name
        echo -e "${BLUE}Searching HuggingFace for: $input${NC}"
        local search_results
        search_results=$(hf_search_models "$input" 10)
        
        if [[ -z "$search_results" ]]; then
            echo -e "${RED}No models found matching: $input${NC}"
            return 1
        fi
        
        # Find a repo with GGUF files
        local found=false
        while IFS= read -r candidate; do
            local gguf_files
            gguf_files=$(hf_find_gguf "$candidate" 2>/dev/null)
            if [[ -n "$gguf_files" ]]; then
                repo="$candidate"
                echo -e "${GREEN}Found repo: $repo${NC}"
                found=true
                break
            fi
        done <<< "$search_results"
        
        if [[ "$found" != "true" ]]; then
            echo -e "${RED}No GGUF files found for any matching model${NC}"
            echo -e "${YELLOW}Try specifying the repo directly:${NC}"
            echo -e "  $0 --download owner/repo --quant $quant"
            return 1
        fi
    fi
    
    # List available quantizations
    echo -e "\n${BLUE}Available quantizations in $repo:${NC}"
    local quants
    quants=$(hf_list_quants "$repo")
    if [[ -n "$quants" ]]; then
        echo "$quants" | head -15 | while read -r q; do
            [[ "$q" == "$quant" ]] && echo -e "  $q (selected)" || echo "  $q"
        done
    fi
    
    # Find best matching file
    echo -e "\n${BLUE}Finding best GGUF file for quant: $quant${NC}"
    
    # Try exact match first, then fuzzy
    local candidates
    candidates=$(hf_find_gguf "$repo" "$quant")
    
    if [[ -z "$candidates" ]]; then
        echo -e "${RED}No GGUF files found in $repo${NC}"
        return 1
    fi
    
    # Pick the best match for the requested quantization
    local selected=""
    local selected_base=""
    local part_count=1
    local total_parts=1
    # Lowercase via tr for bash 3.2 compatibility (macOS default)
    local quant_lc quant_family_lc quant_base_lc
    quant_lc=$(printf '%s' "$quant" | tr 'A-Z' 'a-z')
    quant_family_lc=$(printf '%s' "${quant%%_*}" | tr 'A-Z' 'a-z')
    quant_base_lc=$(printf '%s' "${quant#*-}" | tr 'A-Z' 'a-z')

    # Priority: exact quant match > same quant family > largest available
    # Use case-insensitive matching since HF filenames are lowercase
    while IFS= read -r f; do
        local f_lc
        f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
        if [[ "$f_lc" =~ [-_]${quant_lc}[-_.] ]] || [[ "$f_lc" =~ [-_]${quant_lc}$ ]] || [[ "$f_lc" == *"${quant_lc}"*.gguf ]]; then
            selected="$f"
            break
        fi
    done <<< "$candidates"

    # Fallback: same quant family (e.g., Q4_K_M -> Q4_K_S)
    if [[ -z "$selected" ]]; then
        while IFS= read -r f; do
            local f_lc
            f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
            if [[ "$f_lc" =~ [-_]${quant_family_lc}[-_] ]]; then
                selected="$f"
                break
            fi
        done <<< "$candidates"
    fi

    # Fallback: largest file
    if [[ -z "$selected" ]]; then
        selected=$(echo "$candidates" | head -1)
    fi

    if [[ -z "$selected" ]]; then
        echo -e "${RED}Could not determine filename${NC}"
        return 1
    fi

    # Detect multi-part files: base-00001-of-00003.gguf pattern
    # Use non-greedy .*? to avoid consuming part numbers
    if [[ "$selected" =~ ^(.*?)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
        selected_base="${BASH_REMATCH[1]}"
        part_count="${BASH_REMATCH[2]}"
        total_parts="${BASH_REMATCH[3]}"
        echo -e "${BLUE}Detected multi-part file ($part_count of $total_parts)${NC}"
    fi

    # Collect all files to download
    local all_files=()
    all_files+=("$selected")

    if [[ -n "$selected_base" ]]; then
        while IFS= read -r f; do
            [[ " ${all_files[*]} " == *" $f "* ]] && continue
            # Bash 3.2 safe: escape slashes via tr instead of ${var//pat/repl}
            local escaped_base
            escaped_base=$(printf '%s' "$selected_base" | tr '/' '\\/')
            if [[ "$f" =~ ^${escaped_base}-[0-9]+-of-[0-9]+\.gguf$ ]]; then
                all_files+=("$f")
            fi
        done <<< "$candidates"
    fi

    # Sort by part number
    IFS=$'\n' sorted_files=($(sort -t'_' -k2 -V <<< "${all_files[*]}")); unset IFS

    mkdir -p "$target_dir"

    echo -e "\n${BLUE}Model Download Information${NC}"
    echo ""
    echo -e "  ${GREEN}Repo:${NC}     $repo"
    echo -e "  ${GREEN}Parts:${NC}    ${#sorted_files[@]} file(s) to download"
    for f in "${sorted_files[@]}"; do
        echo -e "    - $f"
    done
    echo -e "  ${GREEN}Quant:${NC}    $quant (requested)"
    echo -e "  ${GREEN}Target:${NC}   $target_dir"
    echo ""

    # Check which files already exist. A filename alone is not enough: an
    # interrupted download can leave a truncated GGUF that the loader rejects.
    local files_to_download=()
    local expected_sizes=()
    local repo_metadata=""
    local i

    repo_metadata=$(hf_api_get "models/$repo?blobs=true" 2>/dev/null || true)
    for ((i = 0; i < ${#sorted_files[@]}; i++)); do
        local f="${sorted_files[$i]}"
        local file_info=""
        local expected_size=""

        if [[ -n "$repo_metadata" ]]; then
            file_info=$(printf '%s' "$repo_metadata" | hf_file_info "$f" || true)
        fi
        if [[ -n "$file_info" ]]; then
            expected_size=$(printf '%s' "$file_info" | jq -r '.size // .lfs.size // empty' 2>/dev/null || true)
        fi
        if [[ ! "$expected_size" =~ ^[0-9]+$ ]]; then
            expected_size=""
        fi
        expected_sizes[$i]="$expected_size"

        if [[ -f "$target_dir/$f" ]]; then
            local local_size
            local_size=$(stat -c %s "$target_dir/$f" 2>/dev/null || stat -f %z "$target_dir/$f")
            if [[ -n "$expected_size" ]] && [[ "$local_size" != "$expected_size" ]]; then
                echo -e "${YELLOW}Cached file has wrong size: $f${NC}"
                echo -e "${YELLOW}  Local: $local_size bytes; expected: $expected_size bytes. Redownloading.${NC}"
                rm -f -- "$target_dir/$f"
                files_to_download+=("$f")
            else
                echo -e "${GREEN}Already exists: $f${NC}"
            fi
        else
            files_to_download+=("$f")
        fi
    done

    if [[ ${#files_to_download[@]} -eq 0 ]]; then
        echo -e "${GREEN}All files already cached.${NC}"
        return 0
    fi

    echo -e "${BLUE}Downloading ${#files_to_download[@]} file(s)...${NC}"
    echo ""

    # Try hf (recommended) or huggingface-cli for downloads
    local use_python=false
    local download_tool=""
    local idx=0
    
    if command -v hf &>/dev/null; then
        download_tool="hf"
    elif command -v huggingface-cli &>/dev/null; then
        download_tool="huggingface-cli"
    else
        use_python=true
    fi
    
    if [[ "$download_tool" == "hf" ]]; then
        # hf is the recommended tool
        for f in "${files_to_download[@]}"; do
            idx=$((idx + 1))
            echo -e "${BLUE}[$idx/${#files_to_download[@]}] $f${NC}"
            if hf download "$repo" "$f" --local-dir "$target_dir" 2>&1; then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                use_python=true
                break
            fi
        done
    elif [[ "$download_tool" == "huggingface-cli" ]]; then
        echo -e "${YELLOW}Note: huggingface-cli is deprecated. Consider installing 'hf' instead.${NC}"
        for f in "${files_to_download[@]}"; do
            idx=$((idx + 1))
            echo -e "${BLUE}[$idx/${#files_to_download[@]}] $f${NC}"
            if huggingface-cli download "$repo" "$f" \
                --local-dir "$target_dir" \
                --local-dir-use-symlinks False 2>&1; then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                use_python=true
                break
            fi
        done
    fi

    # Fallback to Python API if huggingface-cli not available or failed
    if [[ "$use_python" == "true" ]]; then
        if ! python3 -c "import huggingface_hub" 2>/dev/null; then
            echo -e "${RED}huggingface_hub not available. Cannot download.${NC}"
            return 1
        fi

        local idx=0
        for f in "${files_to_download[@]}"; do
            idx=$((idx + 1))
            echo -e "${BLUE}[$idx/${#files_to_download[@]}] $f${NC}"

            if python3 << PYEOF
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id='$repo',
        filename='$f',
        local_dir='$target_dir',
        local_dir_use_symlinks=False
    )
    print(f'  Downloaded to: {path}')
except Exception as e:
    print(f'  Error: {e}')
    exit(1)
PYEOF
            then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                echo -e "${RED}  Failed: $f${NC}"
            fi
        done
    fi

    # Verify every selected shard, including files that were already cached.
    local invalid=0
    for ((i = 0; i < ${#sorted_files[@]}; i++)); do
        local f="${sorted_files[$i]}"
        local expected_size="${expected_sizes[$i]}"
        if [[ ! -f "$target_dir/$f" ]]; then
            echo -e "${RED}Missing: $f${NC}"
            invalid=1
            continue
        fi

        if [[ -n "$expected_size" ]]; then
            local local_size
            local_size=$(stat -c %s "$target_dir/$f" 2>/dev/null || stat -f %z "$target_dir/$f")
            if [[ "$local_size" != "$expected_size" ]]; then
                echo -e "${RED}Invalid size: $f ($local_size bytes; expected $expected_size)${NC}"
                invalid=1
            fi
        fi
    done

    if [[ $invalid -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}All files downloaded successfully!${NC}"
        return 0
    else
        return 1
    fi
}

download_usage() {
    cat << USAGE
${BLUE}Download Model from Hugging Face${NC}

${YELLOW}Usage:${NC}
    $0 --download MODEL [--quant QUANTIZATION]

${YELLOW}Arguments:${NC}
    MODEL          Model name or HuggingFace repo (owner/repo)
                   Examples:
                     - "Qwen3.6-35B" (searches by name)
                     - "unsloth/Qwen3.6-35B-A3B-GGUF" (direct repo)
                     - "bartowski/Mistral-7B-GGUF:Q4_K_M" (repo:quant format)
    --quant        Quantization preference (default: Q4_K_M)
                   Options: Q2_K, Q3_K_M, Q4_K_M, Q4_0, Q5_K_M, Q6_K, Q8_0

${YELLOW}How it works:${NC}
    1. If MODEL contains '/', it's treated as a direct HuggingFace repo
    2. Otherwise, searches HuggingFace for matching models with GGUF files
    3. Finds the best GGUF file matching your quantization preference
    4. Downloads using hf, huggingface-cli, or Python huggingface_hub

${YELLOW}Examples:${NC}
    $0 --download Qwen3.6-35B
    $0 --download Qwen3.6-35B --quant Q5_K_M
    $0 --download mistral-small-3-2 --quant Q4_K_M
    $0 --download unsloth/Qwen3.6-35B-A3B-GGUF
    $0 --download bartowski/Mistral-Small-3.1-24B-Instruct-GGUF --quant Q4_K_M
    $0 --download unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q5_K_M

USAGE
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

# Parse download arguments first
DOWNLOAD_MODEL=""
DOWNLOAD_QUANT="Q4_K_M"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --download)
            DOWNLOAD_MODEL="$2"
            shift 2
            ;;
        --quant)
            DOWNLOAD_QUANT="$2"
            shift 2
            ;;
        *) break ;;
    esac
done

if [[ -n "$DOWNLOAD_MODEL" ]]; then
    download_model "$DOWNLOAD_MODEL" "$DOWNLOAD_QUANT"
    exit $?
fi

# Now parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backend) BACKEND="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -a|--alias) MODEL_ALIAS="$2"; shift 2 ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        -c|--ctx-size) CTX_SIZE="$2"; USER_CTX_SIZE=1; shift 2 ;;
        -n|--n-predict) N_PREDICT="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --kv-cache-type)
            KV_CACHE_TYPE_K="$2"; KV_CACHE_TYPE_V="$2"; USER_KV_CACHE_TYPE=1; shift 2 ;;
        --cache-ssd) SSD_PATH="$2"; shift 2 ;;
        --cache-ssd-checkpoints) SSD_CHECKPOINTS="$2"; shift 2 ;;
        --cache-ssd-hot-window) SSD_HOT_WINDOW="$2"; shift 2 ;;
        --cache-ssd-warm-window) SSD_WARM_WINDOW="$2"; shift 2 ;;
        --cache-ssd-max-cold) SSD_MAX_COLD="$2"; shift 2 ;;
        --cache-ssd-page-size) SSD_PAGE_SIZE="$2"; shift 2 ;;
        --cache-ssd-hot-ram) SSD_HOT_RAM="$2"; shift 2 ;;
        --cache-ssd-warm-ram) SSD_WARM_RAM="$2"; shift 2 ;;
        --cache-ssd-cold-maxsize) SSD_COLD_MAX_SIZE="$2"; shift 2 ;;
        --cache-ssd-system-prompts) SSD_SYSTEM_PROMPTS="$2"; shift 2 ;;
        --cache-ssd-no-fsync) SSD_NO_FSYNC="true"; shift 1 ;;
        --cache-ssd-system-max-days) SSD_SYSTEM_MAX_DAYS="$2"; shift 2 ;;
        --no-ssd-cache) _SSD_DISABLE=true; shift ;;
        --prompt-max) PROMPT_MAX="$2"; shift 2 ;;
        --checkpoint-min-step)
            OVERRIDE_CHECKPOINT_EVERY="$2"; shift 2 ;;
        --ctx-checkpoints)
            OVERRIDE_CTX_CHECKPOINTS="$2"; shift 2 ;;
        --cache-ram)
            OVERRIDE_CACHE_RAM="$2"; shift 2 ;;
        --ubatch-size)
            OVERRIDE_UBATCH_SIZE="$2"; shift 2 ;;
        --np)
            OVERRIDE_N_PARALLEL="$2"; shift 2 ;;
        --preserve-reasoning) PRESERVE_REASONING="true"; shift ;;
        --no-preserve-reasoning) PRESERVE_REASONING="false"; shift ;;
        --reasoning-budget) OVERRIDE_REASONING_BUDGET="$2"; shift 2 ;;
        --no-reasoning-budget) OVERRIDE_REASONING_BUDGET="0"; shift ;;
        --hardware-tier)
            case "$2" in
                halo|standard|handheld)
                    LLAMA_HARDWARE_TIER_OVERRIDE="$2"
                    if [[ "$2" == "halo" ]]; then
                        LLAMA_IS_STRIX_HALO_OVERRIDE="1"
                    else
                        LLAMA_IS_STRIX_HALO_OVERRIDE="0"
                    fi
                    shift 2 ;;
                *) echo -e "${RED}Unknown --hardware-tier value: $2 (must be halo/standard/handheld)${NC}"; exit 1 ;;
            esac
            ;;
        --is-strix-halo) LLAMA_IS_STRIX_HALO_OVERRIDE="1"; shift ;;
        --no-strix-halo) LLAMA_IS_STRIX_HALO_OVERRIDE="0"; shift ;;
        --interactive|-i) INTERACTIVE=true; shift ;;
        --server|-s) SERVER_MODE=true; shift ;;
        --fit) OVERRIDE_FIT="on"; shift ;;
        --print-profile) PRINT_PROFILE=true; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --list-models) list_models ;;
        --list-backends) list_backends_v2 ;;
        --download-help) download_usage ;;
        -h|--help) usage ;;
        --) shift; break ;;
        -*) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
        *) [[ -z "$MODEL" ]] && MODEL="$1"; shift ;;
    esac
done


PROMPT="$*"

# =============================================================================
# Resolve model
# =============================================================================

# If MODEL is a file path, use it directly
if [[ -f "$MODEL" ]]; then
    MODEL="$(realpath "$MODEL")"
# If MODEL matches an alias or basename, resolve it
elif [[ -n "$MODEL" ]]; then
    RESOLVE_IDX=0
    RESOLVE_FOUND=-1
    for entry in "${MODELS_NAME[@]}"; do
        if [[ "$entry" == "$MODEL" ]]; then
            RESOLVE_FOUND=$RESOLVE_IDX
            break
        fi
        RESOLVE_IDX=$((RESOLVE_IDX + 1))
    done
    if [[ $RESOLVE_FOUND -ge 0 ]]; then
        MODEL="${MODELS_PATH[$RESOLVE_FOUND]}"
    fi
    unset RESOLVE_IDX RESOLVE_FOUND
fi

if [[ -z "$MODEL" ]]; then
    echo -e "${YELLOW}No model specified. Use --list-models${NC}"; exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}Model not found: $MODEL${NC}"; exit 1
fi

if [[ "$BACKEND" == "auto" ]]; then
    detect_backend  # Sets BACKEND in current shell (not a subshell)
fi

# Apply hardware-tier override (set by --hardware-tier / --is-strix-halo /
# --no-strix-halo) so assign_profile() reads the user-chosen value
# instead of the auto-detected one. detect-gpu.sh runs at startup before
# argument parsing and only sees LLAMA_*_OVERRIDE env vars; CLI flags
# arrive later and we apply them here.
if [[ -n "${LLAMA_HARDWARE_TIER_OVERRIDE:-}" ]]; then
    LLAMA_HARDWARE_TIER="$LLAMA_HARDWARE_TIER_OVERRIDE"
fi
if [[ -n "${LLAMA_IS_STRIX_HALO_OVERRIDE:-}" ]]; then
    LLAMA_IS_STRIX_HALO="$LLAMA_IS_STRIX_HALO_OVERRIDE"
fi

# All models use dynamic profiling based on file characteristics
# Save user-specified KV cache type before assign_profile overwrites it
_user_kv_cache_k="${KV_CACHE_TYPE_K:-}"
_user_kv_cache_v="${KV_CACHE_TYPE_V:-}"
_user_kv_cache_set="${USER_KV_CACHE_TYPE:-}"
assign_profile "$MODEL"
# Restore user-specified KV cache type if provided
if [[ -n "$_user_kv_cache_set" ]]; then
    KV_CACHE_TYPE_K="$_user_kv_cache_k"
    KV_CACHE_TYPE_V="$_user_kv_cache_v"
fi

# Check if this is a Qwen 3.x model and the fixed template exists
# The fixed template fixes a looping issue with Qwen 3.x models
MODEL_FILENAME=$(basename "$MODEL" .gguf)
if echo "$MODEL_FILENAME" | grep -qiE "qwen3(\.|-)?(5|6|)"; then
    FIXED_TEMPLATE="$PROJECT_ROOT/models/qwen3.5-fixed-template.jinja"
    if [[ -f "$FIXED_TEMPLATE" ]]; then
        EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --chat-template-file '$FIXED_TEMPLATE'"
        log_info "Using fixed Qwen 3.x chat template: $FIXED_TEMPLATE"
    fi
fi

# CLI overrides take precedence over profile-set values. We re-parse
# EXTRA_SERVER_ARGS to drop matching tokens, then append the override.
# The sed patterns are anchored to a leading space so we only match whole
# tokens (avoids --cache-ram eating --cache-ram-mlock or similar).
# Use a space between pattern and number so we match "--cache-ram 6144"
# (the value is a separate token, not glued onto the flag name).
# This runs before PRINT_PROFILE so the printed profile reflects overrides.
strip_and_append() {
    local pattern="$1" replacement="$2" var_name="EXTRA_SERVER_ARGS"
    local stripped
    stripped=$(echo "${!var_name}" | sed -E "s/ ${pattern} [0-9]+//g")
    eval "$var_name=\"\$stripped \$replacement\""
}
[[ -n "$OVERRIDE_CHECKPOINT_EVERY"  ]] && strip_and_append --checkpoint-min-step "--checkpoint-min-step $OVERRIDE_CHECKPOINT_EVERY"
[[ -n "$OVERRIDE_CTX_CHECKPOINTS"    ]] && strip_and_append --ctx-checkpoints            "--ctx-checkpoints $OVERRIDE_CTX_CHECKPOINTS"
[[ -n "$OVERRIDE_CACHE_RAM"          ]] && strip_and_append --cache-ram                  "--cache-ram $OVERRIDE_CACHE_RAM"

# =============================================================================

print_profile_summary() {
    local model_name model_bytes
    model_name=$(basename "$MODEL" .gguf)
    model_bytes=${MODEL_BYTES:-0}
    printf '%b─── Profile ──────────────────────────────────────────────%b\n' "$BLUE" "$NC"
    printf '  %-28s %s\n' "Model:"           "$model_name"
    printf '  %-28s %s\n' "Profile:"         "${profile_name:-unknown}"
    printf '  %-28s %s\n' "Model size:"      "$((model_bytes / 1073741824)) GiB"
    printf '  %-28s %s\n' "Backend:"         "${BACKEND:-auto}"
    printf '  %-28s %s\n' "GPU layers:"      "${GPU_LAYERS:-99}"
    printf '  %-28s %s\n' "Context size:"    "${CTX_SIZE}"
    printf '  %-28s %s\n' "Threads:"         "${THREADS}"
    printf '  %-28s %s/%s\n' "KV cache type:"  "${KV_CACHE_TYPE_K}" "${KV_CACHE_TYPE_V}"
    printf '  %-28s %s\n' "Batch/UBatch:"    "${OVERRIDE_BATCH_SIZE:---batch-size 1024 --ubatch-size 512}"
    printf '  %-28s %s\n' "Reasoning:"       "${OVERRIDE_REASONING:-off}"
    printf '  %-28s %s\n' "Cache RAM:"       "$(echo "${EXTRA_SERVER_ARGS:-}" | grep -oP -- '--cache-ram \K[0-9]+' || echo '-')"
    printf '  %-28s %s\n' "SSD cache:"       "$(if [[ "${_SSD_DISABLE:-false}" == "true" ]]; then echo "disabled"; elif [[ -n "${SSD_PATH:-}" ]]; then echo "${SSD_PATH}"; else echo "default"; fi)"
    printf '  %-28s %s\n' "System cache:"    "$(if [[ "${_SSD_DISABLE:-false}" == "true" ]]; then echo "disabled"; elif [[ -n "${SSD_SYSTEM_PROMPTS:-}" ]]; then echo "${SSD_SYSTEM_PROMPTS} entries"; else echo "off"; fi)"
    printf '  %-28s %s\n' "SSD no-fsync:"   "$(if [[ "${SSD_NO_FSYNC:-}" == "true" ]]; then echo "yes"; else echo "no"; fi)"
    printf '  %-28s %s\n' "Mlock:"           "$(if [[ "$(ulimit -l 2>/dev/null)" == "unlimited" ]]; then echo "yes"; else echo "no (limit: $(ulimit -l) KiB)"; fi)"
    printf '%b──────────────────────────────────────────────────────────%b\n' "$BLUE" "$NC"
}

if [[ "$PRINT_PROFILE" == true ]]; then
    # Machine-parseable KEY=VALUE output for scripts like benchmark.sh.
    # The assign_profile "Auto profile:" line goes to stderr, so stdout is clean.
    cat <<PROFILE_EOF
CTX_SIZE=$CTX_SIZE
MODEL_PATH='$MODEL'
MODEL_NAME=$(basename "$MODEL" .gguf)
MODEL_BYTES=$MODEL_BYTES
GPU_LAYERS=$GPU_LAYERS
THREADS=$THREADS
KV_CACHE_TYPE_K=$KV_CACHE_TYPE_K
KV_CACHE_TYPE_V=$KV_CACHE_TYPE_V
OVERRIDE_BATCH_SIZE='${OVERRIDE_BATCH_SIZE:-"--batch-size 1024 --ubatch-size 512"}'
OVERRIDE_REASONING='${OVERRIDE_REASONING:-off}'
OVERRIDE_REASONING_BUDGET='${OVERRIDE_REASONING_BUDGET:-0}'
EXTRA_SERVER_ARGS='${EXTRA_SERVER_ARGS:-}'
PRESERVE_REASONING='${PRESERVE_REASONING:-true}'
SSD_PATH='${SSD_PATH:-}'
SSD_CHECKPOINTS=$SSD_CHECKPOINTS
SSD_HOT_WINDOW=${SSD_HOT_WINDOW:-4096}
SSD_WARM_WINDOW=${SSD_WARM_WINDOW:-}
SSD_MAX_COLD=${SSD_MAX_COLD:-32}
SSD_PAGE_SIZE=${SSD_PAGE_SIZE:-}
SSD_HOT_RAM=${SSD_HOT_RAM:-}
SSD_WARM_RAM=${SSD_WARM_RAM:-}
SSD_SYSTEM_PROMPTS='${SSD_SYSTEM_PROMPTS:-}'
SSD_SYSTEM_MAX_DAYS='${SSD_SYSTEM_MAX_DAYS:-}'
SSD_NO_FSYNC='${SSD_NO_FSYNC:-}'
OVERRIDE_FIT='${OVERRIDE_FIT:-}'
PROFILE_EOF
    exit 0
fi
# Setup backend
# =============================================================================

# Default SSD cache for all non-SSM models (respects user override)
if [[ "${_SSD_DISABLE:-false}" != "true" ]]; then
    [[ -z "$SSD_PATH" ]] && SSD_PATH="$PROJECT_ROOT/kv-cache"
fi
setup_backend_env

# Get binary paths
LLAMA_BIN=$(get_llama_binary cli)
LLAMA_SERVER=$(get_llama_binary server)

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo -e "${RED}Binary not found: $LLAMA_BIN${NC}"; exit 1
fi

echo -e "${BLUE}Using backend: ${GREEN}${BACKEND}${NC}"
echo -e "${BLUE}Binary: ${GREEN}$LLAMA_SERVER${NC}"

# Pre-launch memory sanity check. The model file is mmap'd lazily so the
# virtual size is mostly free, but the resident footprint depends on access
# patterns. For dense models it's the full file; for MoE models it's a
# fraction (active experts only). We warn if the model is clearly too large.
_total_mem_bytes=$(get_total_memory_bytes)
_total_mem_gb=$((_total_mem_bytes / 1024 / 1024 / 1024))
_model_size_gb=$((MODEL_BYTES / 1024 / 1024 / 1024))
if [[ "$_model_size_gb" -gt $((_total_mem_gb - 2)) ]]; then
    # Model is more than ~92% of OS-visible RAM (not VRAM).
    # Skip warning for GPU-offloaded models (layers in VRAM, not OS RAM).
    if [[ "${GPU_LAYERS:-0}" -lt 99 ]]; then
        echo -e "${YELLOW}Warning: model (${_model_size_gb}GB) is close to system RAM (${_total_mem_gb}GB).${NC}"
        if echo "$(basename "$MODEL" .gguf)" | grep -qiE "moe|a3b|a8b|flash|expert"; then
            echo -e "${YELLOW}         This is a MoE model: resident footprint is much smaller than file size.${NC}"
            echo -e "${YELLOW}         Only active experts are loaded; cold/expert pages stay on disk.${NC}"
        else
            echo -e "${YELLOW}         Dense model: full file will be resident. OOM is likely.${NC}"
            echo -e "${YELLOW}         Use a smaller quant (Q4 or Q3) or a smaller model.${NC}"
        fi
    fi
fi

# GPU-visible memory pre-flight. The model must fit in VRAM carveout + GTT
# (GTT pages come on-demand from system RAM). Without enough GPU memory,
# -ngl 99 crashes with "radv/amdgpu: Not enough memory for command
# submission" + vk::DeviceLostError instead of a clear error. RADV on APUs
# reports only 2/3 of (VRAM+GTT) as DEVICE_LOCAL unless
# radv_enable_unified_heap_on_apu is enabled for llama-server (~/.drirc).
if [[ "$BACKEND" == "vulkan" || "$BACKEND" == "rocm" ]] && [[ "${GPU_LAYERS:-99}" -ge 50 ]] && [[ "$MODEL_BYTES" -gt 0 ]]; then
    _gpu_vis_bytes=0
    for _c in /sys/class/drm/card[0-9]/device; do
        [[ -d "$_c" ]] || continue
        _gpu_vis_bytes=$(( $(cat "$_c/mem_info_vram_total" 2>/dev/null || echo 0) + $(cat "$_c/mem_info_gtt_total" 2>/dev/null || echo 0) ))
        break
    done
    _gpu_vis_gb=$(( _gpu_vis_bytes / 1073741824 ))
    if [[ "$_gpu_vis_gb" -gt 0 ]]; then
        _unified_heap=0
        for _d in "$HOME/.drirc" /usr/share/drirc.d/*.conf; do
            if [[ -f "$_d" ]] && grep -q "radv_enable_unified_heap_on_apu" "$_d" 2>/dev/null; then
                _unified_heap=1
            fi
        done
        _gpu_budget_gb=$(( _gpu_vis_gb - 2 ))  # 2 GiB reserve for compute/staging
        if [[ "$BACKEND" == "vulkan" && "$_unified_heap" -eq 0 ]]; then
            # RADV APU 2/3 split: only this much is DEVICE_LOCAL
            _gpu_budget_gb=$(( _gpu_vis_gb * 2 / 3 - 2 ))
        fi
        if [[ "$MODEL_BYTES" -gt $(( _gpu_budget_gb * 1073741824 )) ]] && [[ "$EXTRA_SERVER_ARGS" != *"--cpu-moe"* ]]; then
            # --cpu-moe pins MoE expert weights to host RAM (handheld tier for
            # models > GPU budget) - the GPU only holds attention/embedding,
            # so the full-model-vs-GPU check does not apply.
            log_error "Model ($_model_size_gb GiB) exceeds GPU-visible memory budget (${_gpu_budget_gb} GiB of ${_gpu_vis_gb} GiB VRAM+GTT)."
            log_error "  With -ngl ${GPU_LAYERS} all layers must fit GPU memory; the load crashes with"
            log_error "  'Not enough memory for command submission' / vk::DeviceLostError when it doesn't."
            if [[ "$BACKEND" == "vulkan" && "$_unified_heap" -eq 0 ]]; then
                log_error "  RADV reports only 2/3 of VRAM+GTT as DEVICE_LOCAL on APUs - enable"
                log_error "  radv_enable_unified_heap_on_apu for llama-server in ~/.drirc."
            fi
            log_error "  Fixes: raise GTT (sudo scripts/apply-ttm-kernel-params.sh <GB>), restore the"
            log_error "  BIOS VRAM carveout, or use a smaller quant."
            exit 1
        fi
        log_info "GPU memory check: model ${_model_size_gb} GiB fits ${_gpu_vis_gb} GiB VRAM+GTT (budget ${_gpu_budget_gb} GiB)"
    fi
fi

# Apply backend-specific env (HSA override for ROCm, Metal debug, etc.)
apply_backend_env

setup_performance

MODEL_SIZE=$(du -h "$MODEL" 2>/dev/null | cut -f1)
# When MODEL is a split-GGUF first shard, du shows the header shard
# size (e.g. 5.7M) while MODEL_BYTES from assign_profile has the
# summed size of all shards. Fix the display.
if [[ "${MODEL_SIZE: -1}" == "M" && "$MODEL_BYTES" -gt 1073741824 ]]; then
    MODEL_SIZE="$((MODEL_BYTES / 1073741824))G"
fi
# MODEL_BYTES is already set by assign_profile; do not recalculate.
MODEL_NAME=$(basename "$MODEL" .gguf)
echo -e "${BLUE}Model: ${GREEN}$MODEL_NAME${NC} ($MODEL_SIZE)"

# =============================================================================
# Fit mode: auto-calculate GPU layers to fit available VRAM
# =============================================================================

# Only pass --fit when the user asked for it; don't inject --fit off by default
if [[ "$OVERRIDE_FIT" == "on" ]]; then
    GPU_LAYERS=-1
    EXTRA_SERVER_ARGS+=" --fit on"
fi

# Build args
# =============================================================================

COMMON_ARGS="-m '$MODEL'"
[[ -n "$MODEL_ALIAS" ]] && COMMON_ARGS="$COMMON_ARGS -a '$MODEL_ALIAS'"
MODEL_BYTES=${MODEL_BYTES:-0}
MEMLOCK_LIMIT_KB=$(ulimit -l 2>/dev/null || true)
if [[ "$MEMLOCK_LIMIT_KB" == "unlimited" || -z "$MEMLOCK_LIMIT_KB" ]]; then
    # mlock not enforced; treat as 0 so we don't mlock
    MEMLOCK_LIMIT_KB=0
fi
MEMLOCK_LIMIT_BYTES=$((MEMLOCK_LIMIT_KB * 1024))
if [[ "$MODEL_BYTES" -gt 0 && "$MODEL_BYTES" -gt "$MEMLOCK_LIMIT_BYTES" ]]; then
    log_info "mlock disabled: model ($((MODEL_BYTES / 1048576)) MiB) larger than memlock limit ($((MEMLOCK_LIMIT_BYTES / 1048576)) MiB)"
    # default is --load-mode mmap, nothing to add
else
    COMMON_ARGS="$COMMON_ARGS --load-mode mlock"
fi
COMMON_ARGS="$COMMON_ARGS -c $CTX_SIZE --threads $THREADS --threads-batch $THREADS"
COMMON_ARGS="$COMMON_ARGS ${OVERRIDE_BATCH_SIZE:---batch-size 1024 --ubatch-size 512} -ngl $GPU_LAYERS"
if [[ -n "$OVERRIDE_UBATCH_SIZE" ]]; then
    COMMON_ARGS=$(echo "$COMMON_ARGS" | sed -E 's/--ubatch-size [0-9]+/--ubatch-size '"$OVERRIDE_UBATCH_SIZE"'/')
fi
COMMON_ARGS="$COMMON_ARGS --cache-type-k $KV_CACHE_TYPE_K --cache-type-v $KV_CACHE_TYPE_V"
[[ -n "$EXTRA_COMMON_ARGS" ]] && COMMON_ARGS="$COMMON_ARGS $EXTRA_COMMON_ARGS"

# KV cache directory for persisting prompt state across restarts
KV_CACHE_DIR="$PROJECT_ROOT/kv-cache"
mkdir -p "$KV_CACHE_DIR"

SERVER_ARGS="--host $HOST --port $PORT"
SERVER_ARGS="$SERVER_ARGS -fa on --jinja"
SERVER_ARGS="$SERVER_ARGS --reasoning ${OVERRIDE_REASONING:-off}"
# Cap reasoning tokens to prevent think loops (disabled for SSM models that don't think)
[[ -n "$OVERRIDE_REASONING_BUDGET" && "$OVERRIDE_REASONING_BUDGET" != "0" ]] && SERVER_ARGS="$SERVER_ARGS --reasoning-budget $OVERRIDE_REASONING_BUDGET"
SERVER_ARGS="$SERVER_ARGS -np ${OVERRIDE_N_PARALLEL:-1} --prio 3 --prio-batch 3 --metrics"
# Checkpoint capacity
SERVER_ARGS="$SERVER_ARGS -ctxcp 64"
# RAM cache reuse threshold
SERVER_ARGS="$SERVER_ARGS --cache-reuse 512"
# Persist KV cache to disk for faster restart (avoids OOM by writing async)
SERVER_ARGS="$SERVER_ARGS --slot-save-path $KV_CACHE_DIR"
# Higher similarity threshold for confident cache matches, unified KV
SERVER_ARGS="$SERVER_ARGS --slot-prompt-similarity 0.20 --kv-unified"

# Append the profile-set + override-merged EXTRA_SERVER_ARGS.
# CLI overrides are applied earlier (before --print-profile) so the printed
# profile reflects them.
[[ -n "$EXTRA_SERVER_ARGS" ]] && SERVER_ARGS="$SERVER_ARGS $EXTRA_SERVER_ARGS"

# Preserve reasoning/thinking in prior assistant messages
# Default: off (the agentic harness preserves knowledge, reasoning in context is redundant)
if [[ "$PRESERVE_REASONING" == "true" ]]; then
    SERVER_ARGS="$SERVER_ARGS --chat-template-kwargs '{\"preserve_thinking\":true}'"
fi



# SSD-backed KV cache
if [[ -n "$SSD_PATH" && "${_SSD_DISABLE:-false}" != "true" ]]; then
    mkdir -p "$SSD_PATH"
    SERVER_ARGS="$SERVER_ARGS --cache-ssd $SSD_PATH"
    [[ -n "$SSD_CHECKPOINTS" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-checkpoints $SSD_CHECKPOINTS"
    [[ -n "$SSD_HOT_WINDOW" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-hot-window $SSD_HOT_WINDOW"
    [[ -n "$SSD_WARM_WINDOW" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-warm-window $SSD_WARM_WINDOW"
    [[ -n "$SSD_MAX_COLD" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-max-cold $SSD_MAX_COLD"
    [[ -n "$SSD_PAGE_SIZE" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-page-size $SSD_PAGE_SIZE"
    [[ -n "$SSD_HOT_RAM" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-hot-ram $SSD_HOT_RAM"
    [[ -n "$SSD_WARM_RAM" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-warm-ram $SSD_WARM_RAM"
    [[ -n "${SSD_COLD_MAX_SIZE:-}" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-cold-maxsize $SSD_COLD_MAX_SIZE"
    [[ -n "${SSD_SYSTEM_PROMPTS:-}" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-system-prompts $SSD_SYSTEM_PROMPTS"
    [[ -n "$SSD_NO_FSYNC" && "$SSD_NO_FSYNC" == "true" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-no-fsync"
    [[ -n "${SSD_SYSTEM_MAX_DAYS:-}" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-system-max-days $SSD_SYSTEM_MAX_DAYS"
    [[ -n "$PROMPT_MAX" && "$PROMPT_MAX" != "0" ]] && SERVER_ARGS="$SERVER_ARGS --prompt-max $PROMPT_MAX"
fi

# =============================================================================
# Kill existing server (ensure only one running)
# =============================================================================

kill_existing_server() {
    local port="$1"
    # Kill any llama-server processes (including ones bound to this port)
    pkill -9 llama-server 2>/dev/null || true
    # Also kill any process holding our port (lsof works on macOS + Linux)
    if command -v lsof &>/dev/null; then
        local pids
        pids=$(lsof -ti tcp:"$port" 2>/dev/null) || true
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    fi
    sleep 1
}

# =============================================================================
# Execute (no sudo needed - runs as local user)
# =============================================================================

# Build environment inline for direct execution (no sudo)
# Note: we use eval here because we need to set env vars (ROCM_PATH, LD_LIBRARY_PATH,
# etc.) in the same command line as launching the binary. A simple `env` prefix would
# re-define the env, but the values are computed by setup_*_env functions and need
# to be quoted to survive paths with spaces. eval is the simplest way to splice them
# in front of the binary invocation without breaking word splitting on $COMMON_ARGS.
EXEC_ENV=""
if [[ "$BACKEND" == "rocm" ]]; then
    EXEC_ENV="ROCM_PATH='$ROCM_PATH' HIP_PATH='$HIP_PATH' HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION='$HSA_OVERRIDE_GFX_VERSION' LD_LIBRARY_PATH='$LD_LIBRARY_PATH'"
elif [[ "$BACKEND" == "vulkan" ]]; then
    EXEC_ENV="LD_LIBRARY_PATH='$LD_LIBRARY_PATH'"
elif [[ "$BACKEND" == "metal" ]]; then
    # Metal needs no env vars at runtime; env is set in-process above
    EXEC_ENV=""
fi

if [[ "$SERVER_MODE" == true ]]; then
    kill_existing_server "$PORT"
    print_profile_summary
    log_info "Server args: ${EXTRA_SERVER_ARGS}"
    echo -e "${BLUE}Starting server on ${HOST}:${PORT}...${NC}"
    eval "$EXEC_ENV" "$LLAMA_SERVER" $COMMON_ARGS $SERVER_ARGS
else
    if [[ "$INTERACTIVE" == true ]]; then
        eval "$EXEC_ENV" "$LLAMA_BIN" $COMMON_ARGS -i
    else
        eval "$EXEC_ENV" "$LLAMA_BIN" $COMMON_ARGS -n $N_PREDICT "$PROMPT"
    fi
fi

