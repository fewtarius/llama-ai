#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
#
# Llama.cpp Unified Runner — Solver-only, fast-fail configuration
# =============================================================================
# Auto-scans ./models for available GGUF files
# Supports Vulkan, ROCm/HIP, Metal, CPU backends
# Uses the optimistic-first solver from scripts/optimize.sh.
# If no configuration fits, exits immediately instead of falling back.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Source required helpers
# -----------------------------------------------------------------------------
source "$PROJECT_ROOT/scripts/detect-gpu.sh"
source "$PROJECT_ROOT/scripts/optimize.sh"

# -----------------------------------------------------------------------------
# Centralised defaults (all overridable via environment)
# -----------------------------------------------------------------------------
: "${ENABLE_CONTEXT_SHIFT:=1}"
: "${KV_CACHE_K_OVERRIDE:=}"
: "${KV_CACHE_V_OVERRIDE:=}"
: "${MOE_UBATCH_OVERRIDE:=}"
: "${CACHE_RAM_OVERRIDE:=}"
: "${LLAMA_THREADS:=}"
: "${LLAMA_GPU_LAYERS:=99}"
: "${LLAMA_CTX_SIZE:=65536}"
: "${LLAMA_N_PREDICT:=256}"
: "${LLAMA_KV_CACHE_TYPE_K:=q8_0}"
: "${LLAMA_KV_CACHE_TYPE_V:=q8_0}"
: "${LLAMA_PORT:=9090}"
: "${LLAMA_HOST:=0.0.0.0}"
: "${LLAMA_SSD_HOT_WINDOW:=4096}"
: "${LLAMA_SSD_WARM_WINDOW:=8192}"
: "${LLAMA_SSD_HOT_RAM:=960}"
: "${LLAMA_SSD_WARM_RAM:=1440}"
: "${LLAMA_SSD_MAX_COLD:=32}"
: "${LLAMA_SSD_CHECKPOINTS:=64}"
: "${LLAMA_SSD_SYSTEM_PROMPTS:=8}"
: "${LLAMA_SSD_SYSTEM_MAX_DAYS:=30}"
: "${LLAMA_PROMPT_MAX:=8}"
: "${LLAMA_REASONING_BUDGET:=4096}"
: "${LLAMA_PRESERVE_REASONING:=true}"
: "${LLAMA_TEMP:=1.0}"
: "${LLAMA_TOP_P:=0.95}"
: "${LLAMA_TOP_K:=20}"
: "${LLAMA_MIN_P:=0.00}"
: "${LLAMA_REPEAT_PENALTY:=1.0}"
: "${LLAMA_PRESENCE_PENALTY:=0.0}"
: "${LLAMA_SLOT_PROMPT_SIMILARITY:=0.20}"
: "${LLAMA_PRIO:=3}"
: "${LLAMA_PRIO_BATCH:=3}"
: "${LLAMA_CTXCP:=64}"
: "${LLAMA_CACHE_REUSE:=512}"
: "${LLAMA_FLASH_ATTN:=auto}"
: "${LLAMA_LOAD_MODE:=dio}"
: "${LLAMA_SSD_NO_FSYNC:=false}"
: "${LLAMA_FIT:=off}"

# Speculative decoding tuning
: "${LLAMA_SPEC_DRAFT_N_MAX:=}"
: "${LLAMA_SPEC_DRAFT_N_MAX_DSPARK:=${LLAMA_SPEC_DRAFT_N_MAX:-3}}"
: "${LLAMA_SPEC_DRAFT_N_MAX_MTP:=${LLAMA_SPEC_DRAFT_N_MAX:-2}}"
: "${LLAMA_SPEC_DRAFT_N_MAX_DFLASH:=${LLAMA_SPEC_DRAFT_N_MAX:-7}}"

# Platform detection
if [[ "$(uname -s)" == "Darwin" ]]; then
    IS_DARWIN=true
    [[ "$(uname -m)" == "arm64" ]] && IS_DARWIN_ARM=true || IS_DARWIN_ARM=false
else
    IS_DARWIN=false; IS_DARWIN_ARM=false
fi

MODEL_DIR="$PROJECT_ROOT/models"

# -----------------------------------------------------------------------------
# Colors & logging
# -----------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$1"; }
log_ok()    { printf '%b[OK]%b   %s\n' "$GREEN" "$NC" "$1"; }
log_warn()  { printf '%b[WARN]%b  %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1"; }

# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------
get_backend_binary() {
    local backend="$1"
    case "$backend" in
        rocm)   echo "$PROJECT_ROOT/src/cachy-llama-rocm/build" ;;
        vulkan) echo "$PROJECT_ROOT/src/cachy-llama-vulkan/build" ;;
        metal)  echo "$PROJECT_ROOT/src/cachy-llama-metal/build" ;;
        cpu)    echo "$PROJECT_ROOT/src/cachy-llama-vulkan/build" ;;
        auto)
            for b in metal rocm vulkan; do
                local dir="$PROJECT_ROOT/src/cachy-llama-$b/build"
                [[ -x "$dir/bin/llama-server" ]] && echo "$dir" && return
            done
            echo ""
            ;;
        *) echo "" ;;
    esac
}

detect_backend() {
    [[ "$BACKEND" != "auto" ]] && return 0
    if [[ "$(uname -s)" == "Darwin" ]] && [[ -x "$PROJECT_ROOT/src/cachy-llama-metal/build/bin/llama-server" ]]; then
        BACKEND="metal"; return 0
    fi
    if [[ -x "$PROJECT_ROOT/src/cachy-llama-vulkan/build/bin/llama-server" ]]; then
        BACKEND="vulkan"; return 0
    fi
    if [[ -x "$PROJECT_ROOT/src/cachy-llama-rocm/build/bin/llama-server" ]]; then
        BACKEND="rocm"; return 0
    fi
    BACKEND="cpu"
}

setup_backend_env() {
    [[ -f "$PROJECT_ROOT/scripts/env.sh" ]] && source "$PROJECT_ROOT/scripts/env.sh" "$BACKEND"
}

get_llama_binary() {
    local cmd="$1"
    echo "$(get_backend_binary "$BACKEND")/bin/llama-$cmd"
}

# -----------------------------------------------------------------------------
# Memory / hardware detection
# -----------------------------------------------------------------------------
get_total_memory_bytes() {
    if $IS_DARWIN; then
        sysctl -n hw.memsize 2>/dev/null || echo 0
    else
        awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

get_available_memory_bytes() {
    if $IS_DARWIN; then
        local ps=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
        local out=$(vm_stat 2>/dev/null)
        local free=$(echo "$out" | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
        local inactive=$(echo "$out" | awk '/Pages inactive/ {gsub(/\./,"",$3); print $3}')
        echo $(( (${free:-0} + ${inactive:-0}) * ps ))
    else
        awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

_get_physical_cores() {
    local cores
    if command -v lscpu &>/dev/null; then
        cores=$(lscpu -p | grep -E '^[0-9]' | cut -d',' -f2 | sort -u | wc -l)
    elif $IS_DARWIN; then
        cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)
    else
        cores=$(nproc)
        if [[ -f /sys/devices/system/cpu/smt/active ]] && [[ "$(cat /sys/devices/system/cpu/smt/active)" == "1" ]]; then
            cores=$((cores / 2))
        fi
    fi
    echo "${cores:-1}"
}

_detect_gpu_budget() {
    local budget=0
    if [[ "$BACKEND" =~ ^(vulkan|rocm|metal)$ ]]; then
        for c in /sys/class/drm/card[0-9]/device; do
            [[ -d "$c" ]] || continue
            local vram=$(cat "$c/mem_info_vram_total" 2>/dev/null || echo 0)
            local gtt=$(cat "$c/mem_info_gtt_total" 2>/dev/null || echo 0)
            budget=$((vram + gtt - 2*1024*1024*1024))
            break
        done
        if $IS_DARWIN && [[ $budget -eq 0 ]]; then
            budget=$(($(get_total_memory_bytes) - 4*1024*1024*1024))
        fi
    fi
    echo "$budget"
}

_compute_gpu_budget_bytes() {
    local vram=0 gtt=0
    if [[ "$BACKEND" =~ ^(vulkan|rocm|metal)$ ]]; then
        for c in /sys/class/drm/card[0-9]/device; do
            [[ -d "$c" ]] || continue
            vram=$(cat "$c/mem_info_vram_total" 2>/dev/null || echo 0)
            gtt=$(cat "$c/mem_info_gtt_total" 2>/dev/null || echo 0)
            break
        done
    fi

    local gpu_vis_bytes=$(( vram + gtt ))
    local gpu_vis_gib=$(( gpu_vis_bytes / 1073741824 ))

    local total_ram_gib=$(( $(get_total_memory_bytes) / 1073741824 ))
    [[ $total_ram_gib -le 0 ]] && total_ram_gib=1

    local os_reserve_gib=8
    if [[ $total_ram_gib -le 16 ]]; then
        os_reserve_gib=$(( total_ram_gib / 4 ))
    fi

    local unified_budget_gib=$(( gpu_vis_gib - 2 ))
    local system_limit_gib=$(( total_ram_gib - os_reserve_gib ))
    [[ $unified_budget_gib -gt $system_limit_gib ]] && unified_budget_gib=$system_limit_gib
    [[ $unified_budget_gib -lt 1 ]] && unified_budget_gib=1

    echo $(( unified_budget_gib * 1073741824 ))
}

# -----------------------------------------------------------------------------
# Model discovery & download
# -----------------------------------------------------------------------------
declare -a MODELS_NAME=()
declare -a MODELS_PATH=()

scan_models() {
    MODELS_NAME=(); MODELS_PATH=()
    [[ ! -d "$MODEL_DIR" ]] && return
    while IFS= read -r -d '' f; do
        MODELS_NAME+=("$(basename "$f" .gguf)")
        MODELS_PATH+=("$f")
    done < <(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" -print0 2>/dev/null)
}
scan_models

list_models() {
    echo -e "${BLUE}Available Models:${NC}"
    for i in $(printf '%s\n' "${!MODELS_NAME[@]}" | sort -n); do
        echo -e "  ${GREEN}${MODELS_NAME[$i]}${NC}  - $(du -h "${MODELS_PATH[$i]}" 2>/dev/null | cut -f1)"
    done
    [[ ${#MODELS_NAME[@]} -eq 0 ]] && echo -e "  ${YELLOW}No models found in $MODEL_DIR${NC}"
    exit 0
}

list_backends_v2() {
    echo -e "${BLUE}Available Backends:${NC}"
    for b in rocm vulkan metal; do
        local bin="$PROJECT_ROOT/src/cachy-llama-$b/build/bin/llama-cli"
        if [[ -x "$bin" ]]; then
            echo -e "  ${GREEN}[*] ${b^}${NC}       - available"
        else
            echo -e "  ${YELLOW}[ ] ${b^}${NC}       - not built"
        fi
    done
    echo -e "  ${GREEN}[*] CPU${NC}         - always available"
    exit 0
}

hf_api_get() { curl -s -L --fail "https://huggingface.co/api/$1" 2>/dev/null; }
hf_search_models() { hf_api_get "models?search=${1// /+}&sort=downloads&direction=-1&limit=${2:-10}" | jq -r '.[].id // empty' 2>/dev/null; }
hf_list_files() { hf_api_get "models/$1" | jq -r '.siblings[].rfilename // empty' 2>/dev/null; }
hf_file_info() { jq -c --arg filename "$1" 'first(.siblings[]? | select(.rfilename == $filename)) // empty' 2>/dev/null; }

hf_find_gguf() {
    local repo="$1"
    local quant_pattern="${2:-Q4_K_M}"
    local files
    files=$(hf_list_files "$repo") || return 1
    local quant_base="${quant_pattern#*-}"
    local quant_family="${quant_base%%_*}"
    local quant_lc quant_base_lc quant_family_lc
    quant_lc=$(printf '%s' "$quant_pattern" | tr 'A-Z' 'a-z')
    quant_base_lc=$(printf '%s' "$quant_base" | tr 'A-Z' 'a-z')
    quant_family_lc=$(printf '%s' "$quant_family" | tr 'A-Z' 'a-z')
    local matches=()
    while IFS= read -r f; do
        [[ "$f" == *.gguf ]] || continue
        local f_lc
        f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
        if [[ "$f_lc" =~ [-_]${quant_lc}[-_.] ]] || \
           [[ "$f_lc" =~ [-_]${quant_lc}$ ]] || \
           [[ "$f_lc" =~ [-_]${quant_base_lc}[-_.] ]] || \
           [[ "$f_lc" =~ [-_]${quant_base_lc}$ ]] || \
           [[ "$f_lc" =~ [-_]${quant_family_lc}[-_] ]]; then
            matches+=("$f")
        fi
    done <<< "$files"
    if [[ ${#matches[@]} -eq 0 ]]; then
        while IFS= read -r f; do
            [[ "$f" == *.gguf ]] && matches+=("$f")
        done <<< "$files"
    fi
    printf '%s\n' "${matches[@]}" | sort -t'_' -k2 -V | tac
}

hf_list_quants() {
    local repo="$1"
    local files
    files=$(hf_list_files "$repo") 2>/dev/null || return 1
    echo "$files" | grep -oE '[Qq][0-9]+[_-]?K?_?[SMXL]?' | sort -u | head -20
}

download_model() {
    local input="$1"
    local quant="${2:-Q4_K_M}"
    local target_dir="$MODEL_DIR"
    local repo=""
    if [[ "$input" == *"/"* ]]; then
        repo="${input%%:*}"
        if [[ "$input" == *":"* ]]; then
            quant="${input##*:}"
        fi
    else
        echo -e "${BLUE}Searching HuggingFace for: $input${NC}"
        local search_results
        search_results=$(hf_search_models "$input" 10)
        if [[ -z "$search_results" ]]; then
            echo -e "${RED}No models found matching: $input${NC}"
            return 1
        fi
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
            return 1
        fi
    fi

    echo -e "\n${BLUE}Available quantizations in $repo:${NC}"
    local quants
    quants=$(hf_list_quants "$repo")
    if [[ -n "$quants" ]]; then
        echo "$quants" | head -15 | while read -r q; do
            [[ "$q" == "$quant" ]] && echo -e "  $q (selected)" || echo "  $q"
        done
    fi

    echo -e "\n${BLUE}Finding best GGUF file for quant: $quant${NC}"
    local candidates
    candidates=$(hf_find_gguf "$repo" "$quant")
    if [[ -z "$candidates" ]]; then
        echo -e "${RED}No GGUF files found in $repo${NC}"
        return 1
    fi

    local selected=""
    local selected_base=""
    local part_count=1
    local total_parts=1
    local quant_lc quant_family_lc quant_base_lc
    quant_lc=$(printf '%s' "$quant" | tr 'A-Z' 'a-z')
    quant_family_lc=$(printf '%s' "${quant%%_*}" | tr 'A-Z' 'a-z')
    quant_base_lc=$(printf '%s' "${quant#*-}" | tr 'A-Z' 'a-z')

    while IFS= read -r f; do
        local f_lc
        f_lc=$(printf '%s' "$f" | tr 'A-Z' 'a-z')
        if [[ "$f_lc" =~ [-_]${quant_lc}[-_.] ]] || [[ "$f_lc" =~ [-_]${quant_lc}$ ]] || [[ "$f_lc" == *"${quant_lc}"*.gguf ]]; then
            selected="$f"
            break
        fi
    done <<< "$candidates"

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

    if [[ -z "$selected" ]]; then
        selected=$(echo "$candidates" | head -1)
    fi

    if [[ -z "$selected" ]]; then
        echo -e "${RED}Could not determine filename${NC}"
        return 1
    fi

    if [[ "$selected" =~ ^(.*?)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
        selected_base="${BASH_REMATCH[1]}"
        part_count="${BASH_REMATCH[2]}"
        total_parts="${BASH_REMATCH[3]}"
        echo -e "${BLUE}Detected multi-part file ($part_count of $total_parts)${NC}"
    fi

    local all_files=()
    all_files+=("$selected")
    if [[ -n "$selected_base" ]]; then
        while IFS= read -r f; do
            [[ " ${all_files[*]} " == *" $f "* ]] && continue
            local escaped_base
            escaped_base=$(printf '%s' "$selected_base" | tr '/' '\\/')
            if [[ "$f" =~ ^${escaped_base}-[0-9]+-of-[0-9]+\.gguf$ ]]; then
                all_files+=("$f")
            fi
        done <<< "$candidates"
    fi

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

    local files_to_download=()
    local expected_sizes=()
    local repo_metadata=""
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
        [[ ! "$expected_size" =~ ^[0-9]+$ ]] && expected_size=""
        expected_sizes[$i]="$expected_size"

        if [[ -f "$target_dir/$f" ]]; then
            local local_size
            local_size=$(stat -c %s "$target_dir/$f" 2>/dev/null || stat -f %z "$target_dir/$f")
            if [[ -n "$expected_size" ]] && [[ "$local_size" != "$expected_size" ]]; then
                echo -e "${YELLOW}Cached file has wrong size: $f${NC}"
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
            if huggingface-cli download "$repo" "$f" --local-dir "$target_dir" --local-dir-use-symlinks False 2>&1; then
                echo -e "${GREEN}  Downloaded: $f${NC}"
            else
                use_python=true
                break
            fi
        done
    fi

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

# -----------------------------------------------------------------------------
# Backend environment setup
# -----------------------------------------------------------------------------
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
    export MESA_SHADER_CACHE_MAX_SIZE="${MESA_SHADER_CACHE_MAX_SIZE:-2G}"
    export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-$HOME/.cache/mesa_shader_cache}"
    mkdir -p "$MESA_SHADER_CACHE_DIR"
    export RADV_PERFTEST="${RADV_PERFTEST:-gplp}"
    if [[ "${LLAMA_GFX_ARCH:-}" == "gfx1103" ]] && [[ -z "${GGML_VK_NODES_PER_SUBMIT:-}" ]]; then
        export GGML_VK_NODES_PER_SUBMIT=64
    fi
}

setup_metal_env() {
    export GGML_METAL_DEVICE_DEBUG=0
}

apply_backend_env() {
    case "$BACKEND" in
        rocm)   setup_rocm_env ;;
        vulkan) setup_vulkan_env ;;
        metal)  setup_metal_env ;;
    esac
}

setup_performance() {
    if command -v cpupower &>/dev/null; then
        sudo cpupower frequency-set -g performance 2>/dev/null || true
    fi
    for p in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do
        echo performance | sudo tee "$p" >/dev/null 2>&1 || true
    done
    for card in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
        if [[ -f "$card" ]]; then
            echo "auto" | sudo tee "$card" >/dev/null 2>&1 || true
        fi
    done
}

restore_performance() {
    for p in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do
        echo balance_performance | sudo tee "$p" >/dev/null 2>&1 || true
    done
}

_cleanup() { restore_performance; }
trap _cleanup EXIT

# -----------------------------------------------------------------------------
# Profile assignment — solver only, fast-fail
# -----------------------------------------------------------------------------
profile_name=""

_scan_gguf_arch() {
    local gguf_path="$1"
    [[ ! -f "$gguf_path" || ! -r "$gguf_path" ]] && return 0
    local tmp=$(mktemp /tmp/llama-scan-XXXXXX)
    dd if="$gguf_path" of="$tmp" bs=16384 count=1 2>/dev/null || { rm -f "$tmp"; return 0; }
    if grep -q 'expert_count' "$tmp" 2>/dev/null; then is_moe=true; fi
    if grep -q 'ssm\.' "$tmp" 2>/dev/null && ! grep -q 'full_attention_interval' "$tmp" 2>/dev/null && [[ "${is_moe:-false}" != "true" ]]; then
        is_ssm=true
    fi
    if grep -q 'nextn_predict_layers' "$tmp" 2>/dev/null; then
        is_mtp=true
    fi
    rm -f "$tmp"
}

_apply_reasoning_defaults() {
    EXTRA_SERVER_ARGS+=" --temp ${LLAMA_TEMP} --top-p ${LLAMA_TOP_P} --top-k ${LLAMA_TOP_K} --min-p ${LLAMA_MIN_P}"
    EXTRA_SERVER_ARGS+=" --repeat-penalty ${LLAMA_REPEAT_PENALTY} --presence-penalty ${LLAMA_PRESENCE_PENALTY}"
    OVERRIDE_REASONING="on"
}

_apply_ssd_defaults() {
    SSD_HOT_WINDOW="${LLAMA_SSD_HOT_WINDOW}"
    SSD_WARM_WINDOW="${LLAMA_SSD_WARM_WINDOW}"
    SSD_HOT_RAM="${LLAMA_SSD_HOT_RAM}"
    SSD_WARM_RAM="${LLAMA_SSD_WARM_RAM}"
    SSD_MAX_COLD="${LLAMA_SSD_MAX_COLD}"
    SSD_CHECKPOINTS="${LLAMA_SSD_CHECKPOINTS}"
    SSD_SYSTEM_PROMPTS="${LLAMA_SSD_SYSTEM_PROMPTS}"
    SSD_SYSTEM_MAX_DAYS="${LLAMA_SSD_SYSTEM_MAX_DAYS}"
    PROMPT_MAX="${LLAMA_PROMPT_MAX}"
}

_detect_draft_model() {
    local model_path="$1" tag_lc="$2"
    [[ -z "$tag_lc" ]] && { echo ""; return; }
    local base
    base=$(basename "$model_path")
    if [[ "$base" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        base="${base%-${BASH_REMATCH[1]}-of-${BASH_REMATCH[2]}.gguf}"
    fi
    base="${base%-*}"
    [[ "$base" == *"-UD" ]] && base="${base%-UD}"
    [[ -z "$base" ]] && { echo ""; return; }
    local base_lc
    base_lc=$(echo "$base" | tr '[:upper:]' '[:lower:]')
    local f fb
    while IFS= read -r -d '' f; do
        fb=$(basename "$f")
        if echo "$fb" | grep -qiE "^${base_lc}-${tag_lc}-[A-Za-z0-9_]+\.gguf$"; then
            echo "$f" && return
        fi
    done < <(find "$MODEL_DIR" -maxdepth 1 -iname "*-${tag_lc}-*.gguf" -print0 2>/dev/null)
    echo ""
}

compute_model_size_bytes() {
    local model_path="$1"
    local filename=$(basename "$model_path")
    if [[ "$model_path" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        local shard_base="${model_path%-${BASH_REMATCH[1]}-of-${BASH_REMATCH[2]}.gguf}"
        local shard_count=$((10#${BASH_REMATCH[2]}))
        local size=0
        for ((i=1; i<=shard_count; i++)); do
            local sf=$(printf "%s-%05d-of-%05d.gguf" "$shard_base" "$i" "$shard_count")
            [[ -f "$sf" ]] && size=$((size + $(stat -c%s "$sf" 2>/dev/null || stat -f%z "$sf" 2>/dev/null || echo 0)))
        done
        [[ $size -eq 0 ]] && size=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null || echo 0)
        echo "$size"
    else
        stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null || echo 0
    fi
}

assign_profile() {
    local model_path="$1"
    local filename=$(basename "$model_path")

    MODEL_BYTES=$(compute_model_size_bytes "$model_path")
    local size_gb=$((MODEL_BYTES / 1073741824))

    is_moe="false"; is_ssm="false"; is_mtp="false"
    if echo "$filename" | grep -qiE "moe|a3b|a8b|flash|expert|gpt-oss"; then
        is_moe=true
    fi
    if echo "$filename" | grep -qiE "ssm|mamba|jamba|falcon-h1|rwkv"; then
        is_ssm=true
    fi
    _scan_gguf_arch "$model_path"

    prof_dspark=$(_detect_draft_model "$model_path" "dspark")
    prof_dflash=$(_detect_draft_model "$model_path" "dflash")

    solve_optimal_config "$model_path"
    apply_user_overrides

    # Fast-fail: solver's absolute fallback indicates no fitting configuration.
    if [[ "${SOLVER_REASONS[*]}" == *"no fit at any strategy/kv/ctx"* ]]; then
        log_error "Solver could not find a configuration that fits within available memory budget."
        log_error "GPU budget: ${GPU_BUDGET_GB:-unknown} GiB, model size: ${size_gb} GiB."
        exit 1
    fi

    CTX_SIZE="${SOLVER_CTX_SIZE}"
    KV_CACHE_TYPE_K="${SOLVER_K_TYPE}"
    KV_CACHE_TYPE_V="${SOLVER_V_TYPE}"
    THREADS="${SOLVER_THREADS}"
    THREADS_BATCH="${SOLVER_THREADS_BATCH}"
    OVERRIDE_BATCH_SIZE="--batch-size ${SOLVER_BATCH} --ubatch-size ${SOLVER_UBATCH}"
    LOAD_MODE="${SOLVER_LOAD_MODE}"

    EXTRA_SERVER_ARGS="--no-mmproj --flash-attn ${LLAMA_FLASH_ATTN}"

    if [[ "${is_moe}" == "true" ]]; then
        case "${SOLVER_MOE_STRATEGY}" in
            cpu)
                local total_mem=$(get_total_memory_bytes)
                if [[ ${MODEL_BYTES:-0} -lt $total_mem ]]; then
                    EXTRA_SERVER_ARGS+=" --cpu-moe --load-mode none"
                else
                    EXTRA_SERVER_ARGS+=" --cpu-moe"
                fi
                ;;
            residency)
                EXTRA_SERVER_ARGS+=" --moe-expert-residency"
                ;;
        esac
    fi

    _apply_reasoning_defaults
    OVERRIDE_REASONING_BUDGET="${SOLVER_REASONING_BUDGET:-$LLAMA_REASONING_BUDGET}"

    # SSD / prompt-cache settings
    if [[ "${SSD_CACHE_DISABLED_USER:-false}" == "true" ]]; then
        _SSD_DISABLE=true
    else
        if [[ "${SOLVER_SSD_ENABLE}" != "true" ]]; then
            _SSD_DISABLE=true
        else
            _SSD_DISABLE=false
            _apply_ssd_defaults
            SSD_HOT_RAM="${SOLVER_SSD_HOT_RAM}"
            SSD_WARM_RAM="${SOLVER_SSD_WARM_RAM}"
        fi
    fi

    # Speculative decoding
    if [[ "${SOLVER_DRAFT_ENABLE}" == "true" && -n "${prof_dspark:-}" ]]; then
        log_info "DSpark speculative decoding enabled: $prof_dspark (n_max=$LLAMA_SPEC_DRAFT_N_MAX_DSPARK)"
        EXTRA_SERVER_ARGS+=" -md $prof_dspark --spec-type draft-dspark --spec-draft-n-max $LLAMA_SPEC_DRAFT_N_MAX_DSPARK --fit off"
    elif [[ "${is_mtp}" == "true" && "${SOLVER_DRAFT_ENABLE}" == "true" ]]; then
        log_info "MTP speculative decoding enabled (n_max=$LLAMA_SPEC_DRAFT_N_MAX_MTP)"
        EXTRA_SERVER_ARGS+=" --spec-type draft-mtp --spec-draft-n-max $LLAMA_SPEC_DRAFT_N_MAX_MTP --fit off"
    elif [[ "${SOLVER_DRAFT_ENABLE}" == "true" && -n "${prof_dflash:-}" ]]; then
        log_info "DFlash speculative decoding enabled: $prof_dflash (n_max=$LLAMA_SPEC_DRAFT_N_MAX_DFLASH)"
        EXTRA_SERVER_ARGS+=" -md $prof_dflash --spec-type draft-dflash --spec-draft-n-max $LLAMA_SPEC_DRAFT_N_MAX_DFLASH --fit off"
    fi

    profile_name="${SOLVER_PROFILE_NAME}"

    SOLVER_CHECKPOINT_MIN="${SOLVER_CHECKPOINT_MIN:-32768}"
    SOLVER_CHECKPOINTS="${SOLVER_CHECKPOINTS:-8}"
    EXTRA_SERVER_ARGS+=" --checkpoint-min-step ${SOLVER_CHECKPOINT_MIN} --ctx-checkpoints ${SOLVER_CHECKPOINTS}"
    EXTRA_SERVER_ARGS+=" --no-checkpoint-near-end"

    local cache_ram_mib="${SOLVER_CACHE_RAM:-0}"
    [[ $cache_ram_mib -gt 0 ]] && EXTRA_SERVER_ARGS+=" --cache-ram $cache_ram_mib"
    EXTRA_SERVER_ARGS+=" --cache-ram $cache_ram_mib"

    log_info "Solver chose: ctx=$CTX_SIZE KV=$KV_CACHE_TYPE_K/$KV_CACHE_TYPE_V ubatch=$SOLVER_UBATCH batch=$SOLVER_BATCH threads=$THREADS_BATCH/$THREADS"
    if [[ ${#SOLVER_REASONS[@]} -gt 0 ]]; then
        log_info "Solver detune steps: ${SOLVER_REASONS[*]}"
    fi
    log_info "Solver overrides applied: ${SOLVER_OVERRIDES[*]:-(none)}"
    printf '%bAuto profile (solver): %b%s%b (%sGB, MoE=%s, SSM=%s)%b\n' \
        "$CYAN" "$GREEN" "$profile_name" "$NC" "$size_gb" "${is_moe:-false}" "${is_ssm:-false}" "$NC"
}

# -----------------------------------------------------------------------------
# Chat template detection
# -----------------------------------------------------------------------------
detect_chat_template() {
    local name="$1"
    local tbase="$PROJECT_ROOT/CachyLLama/models/templates"
    for pair in \
        "deepseek[_-]?v4[_-]?flash:$tbase/deepseek-ai-DeepSeek-V4-Flash-0731.jinja" \
        "glm[_-]?4[.]7[_-]?flash:$tbase/GLM-4.7-Flash.jinja" \
        "qwen3[_-]?coder:$tbase/Qwen3-Coder.jinja" \
        "laguna[_-]?s[_-]?2[.]1:$tbase/poolside-Laguna-S-2.1.jinja"
    do
        local pattern="${pair%%:*}" file="${pair##*:}"
        if echo "$name" | grep -qiE "$pattern"; then
            [[ -f "$file" ]] && echo "$file" && return
        fi
    done
}

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    cat << USAGE
${BLUE}Llama.cpp Runner — Solver-only, fast-fail

${YELLOW}Usage:${NC}
    $0 [options] [model_or_file] [-- server options]
    $0 --download MODEL [--quant QUANT]

${YELLOW}Options:${NC}
    -b, --backend BACKEND    Backend: auto, rocm, vulkan, metal, cpu
    -m, --model MODEL       Model file or alias
    -a, --alias NAME        API model alias
    -t, --threads N         CPU threads (default: auto-detected)
    -c, --ctx-size N        Context size (default: auto per profile)
    -n, --n-predict N       Tokens to generate
    -ngl, --gpu-layers N    GPU layers (default: 99)
    --kv-cache-type TYPE    Force KV cache quantization (e.g. q8_0)
    --interactive           Interactive chat mode
    --server                Run as API server
    --port PORT             Server port (default: 9090)
    --fit                   Auto-fit GPU layers to available VRAM
    --host HOST             Server host (default: 0.0.0.0)
    --list-models           List available models
    --list-backends         List available backends
    --preserve-reasoning    Include reasoning in prior assistant messages
    --no-preserve-reasoning Strip reasoning from prior assistant messages (default)
    --reasoning-budget N    Max thinking tokens per response (default: 4096)
    --no-reasoning-budget   Disable thinking token limit
    --hardware-tier TIER    Override detected tier: halo, standard, handheld
    --is-strix-halo         Force Strix Halo preset
    --no-strix-halo         Force non-Strix-Halo preset
    --no-ssd-cache          Disable SSD KV cache entirely
    -h, --help              Show this help

${YELLOW}Environment overrides:${NC}
    LLAMA_THREADS           Fixed number of CPU threads (overrides auto-detect)
    ENABLE_CONTEXT_SHIFT    0 to disable context shifting (default: 1)
    KV_CACHE_K_OVERRIDE     Force K type (e.g. q8_0)
    KV_CACHE_V_OVERRIDE     Force V type (e.g. q4_0)
    MOE_UBATCH_OVERRIDE     Force ubatch size
    CACHE_RAM_OVERRIDE      Force cache-ram in MiB (disables auto-detection)

${YELLOW}Examples:${NC}
    $0 --server Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003
    $0 --backend vulkan --interactive Qwen3-14B
    $0 --download Qwen3.6-35B

USAGE
    exit 0
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
DOWNLOAD_MODEL=""; DOWNLOAD_QUANT="Q4_K_M"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --download) DOWNLOAD_MODEL="$2"; shift 2 ;;
        --quant) DOWNLOAD_QUANT="$2"; shift 2 ;;
        *) break ;;
    esac
done
if [[ -n "$DOWNLOAD_MODEL" ]]; then
    download_model "$DOWNLOAD_MODEL" "$DOWNLOAD_QUANT"
    exit $?
fi

USER_CTX_SIZE=""
USER_KV_CACHE_TYPE=""
BACKEND="auto"
MODEL=""
MODEL_ALIAS=""
GPU_LAYERS="$LLAMA_GPU_LAYERS"
CTX_SIZE="$LLAMA_CTX_SIZE"
N_PREDICT="$LLAMA_N_PREDICT"
KV_CACHE_TYPE_K="$LLAMA_KV_CACHE_TYPE_K"
KV_CACHE_TYPE_V="$LLAMA_KV_CACHE_TYPE_V"
INTERACTIVE=false
SERVER_MODE=false
PRINT_PROFILE=false
PORT="$LLAMA_PORT"
HOST="$LLAMA_HOST"
OVERRIDE_FIT="$LLAMA_FIT"
OVERRIDE_REASONING_BUDGET="$LLAMA_REASONING_BUDGET"
PRESERVE_REASONING="$LLAMA_PRESERVE_REASONING"
OVERRIDE_CHECKPOINT_EVERY=""
OVERRIDE_CTX_CHECKPOINTS=""
OVERRIDE_CACHE_RAM=""
OVERRIDE_UBATCH_SIZE=""
OVERRIDE_N_PARALLEL=""
SSD_PATH=""
SSD_CHECKPOINTS=""
SSD_HOT_WINDOW=""
SSD_WARM_WINDOW=""
SSD_MAX_COLD=""
SSD_PAGE_SIZE=""
SSD_HOT_RAM=""
SSD_WARM_RAM=""
SSD_COLD_MAX_SIZE=""
SSD_SYSTEM_PROMPTS=""
SSD_NO_FSYNC=""
SSD_SYSTEM_MAX_DAYS=""
PROMPT_MAX=""
EXTRA_COMMON_ARGS=""
EXTRA_SERVER_ARGS=""
OVERRIDE_BATCH_SIZE=""
_SSD_DISABLE=false
SSD_CACHE_DISABLED_USER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backend) BACKEND="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -a|--alias) MODEL_ALIAS="$2"; shift 2 ;;
        -t|--threads) LLAMA_THREADS="$2"; shift 2 ;;
        -c|--ctx-size) CTX_SIZE="$2"; USER_CTX_SIZE=1; shift 2 ;;
        -n|--n-predict) N_PREDICT="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --kv-cache-type) KV_CACHE_TYPE_K="$2"; KV_CACHE_TYPE_V="$2"; USER_KV_CACHE_TYPE=1; shift 2 ;;
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
        --no-ssd-cache) SSD_CACHE_DISABLED_USER=true; _SSD_DISABLE=true; shift ;;
        --prompt-max) PROMPT_MAX="$2"; shift 2 ;;
        --checkpoint-min-step) OVERRIDE_CHECKPOINT_EVERY="$2"; shift 2 ;;
        --ctx-checkpoints) OVERRIDE_CTX_CHECKPOINTS="$2"; shift 2 ;;
        --cache-ram) OVERRIDE_CACHE_RAM="$2"; shift 2 ;;
        --ubatch-size) OVERRIDE_UBATCH_SIZE="$2"; shift 2 ;;
        --np) OVERRIDE_N_PARALLEL="$2"; shift 2 ;;
        --preserve-reasoning) PRESERVE_REASONING="true"; shift ;;
        --no-preserve-reasoning) PRESERVE_REASONING="false"; shift ;;
        --reasoning-budget) OVERRIDE_REASONING_BUDGET="$2"; shift 2 ;;
        --no-reasoning-budget) OVERRIDE_REASONING_BUDGET="0"; shift ;;
        --hardware-tier)
            case "$2" in
                halo|standard|handheld)
                    LLAMA_HARDWARE_TIER_OVERRIDE="$2"
                    if [[ "$2" == "halo" ]]; then LLAMA_IS_STRIX_HALO_OVERRIDE="1"; else LLAMA_IS_STRIX_HALO_OVERRIDE="0"; fi
                    shift 2 ;;
                *) log_error "Unknown --hardware-tier value: $2"; exit 1 ;;
            esac ;;
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
        -h|--help) usage ;;
        --) shift; break ;;
        -*) log_error "Unknown option: $1"; exit 1 ;;
        *) [[ -z "$MODEL" ]] && MODEL="$1" || {
               if [[ "$1" =~ ^[0-9]+$ ]]; then
                   CTX_SIZE="$1"
                   USER_CTX_SIZE=1
                   shift
                   continue
               fi
           }; shift ;;
    esac
done
PROMPT="$*"

# -----------------------------------------------------------------------------
# Resolve model
# -----------------------------------------------------------------------------
if [[ -f "$MODEL" ]]; then
    MODEL="$(realpath "$MODEL")"
elif [[ -n "$MODEL" ]]; then
    for i in "${!MODELS_NAME[@]}"; do
        if [[ "${MODELS_NAME[$i]}" == "$MODEL" ]]; then
            MODEL="${MODELS_PATH[$i]}"
            break
        fi
    done
fi
[[ -z "$MODEL" ]] && { log_error "No model specified."; exit 1; }
[[ ! -f "$MODEL" ]] && { log_error "Model not found: $MODEL"; exit 1; }

# -----------------------------------------------------------------------------
# Backend setup and GPU budget detection
# -----------------------------------------------------------------------------
[[ "$BACKEND" == "auto" ]] && detect_backend

GPU_BUDGET_BYTES="$(_detect_gpu_budget)"
GPU_BUDGET_GB=$((GPU_BUDGET_BYTES / 1073741824))
if [[ $GPU_BUDGET_BYTES -gt 0 ]]; then
    log_info "Detected GPU budget: ${GPU_BUDGET_GB} GiB (VRAM+GTT)"
else
    log_warn "Could not detect GPU budget; cache-ram will use conservative fallback"
fi

SOLVER_GPU_BUDGET_BYTES="$(_compute_gpu_budget_bytes)"
export SOLVER_GPU_BUDGET_BYTES

[[ -n "${LLAMA_HARDWARE_TIER_OVERRIDE:-}" ]] && LLAMA_HARDWARE_TIER="$LLAMA_HARDWARE_TIER_OVERRIDE"
[[ -n "${LLAMA_IS_STRIX_HALO_OVERRIDE:-}" ]] && LLAMA_IS_STRIX_HALO="$LLAMA_IS_STRIX_HALO_OVERRIDE"

# Save user-specified KV cache type before assign_profile overwrites
_user_kv_k="${KV_CACHE_TYPE_K:-}"
_user_kv_v="${KV_CACHE_TYPE_V:-}"
_user_kv_set="${USER_KV_CACHE_TYPE:-}"

assign_profile "$MODEL"

if [[ -n "$_user_kv_set" ]]; then
    KV_CACHE_TYPE_K="$_user_kv_k"
    KV_CACHE_TYPE_V="$_user_kv_v"
fi

# -----------------------------------------------------------------------------
# Chat template detection
# -----------------------------------------------------------------------------
MODEL_FILENAME=$(basename "$MODEL" .gguf)
_CHAT_TEMPLATE=$(detect_chat_template "$MODEL_FILENAME")
if [[ -n "$_CHAT_TEMPLATE" ]]; then
    EXTRA_SERVER_ARGS+=" --chat-template-file '$_CHAT_TEMPLATE'"
    log_info "Using chat template: $_CHAT_TEMPLATE"
fi

# -----------------------------------------------------------------------------
# Override strip-and-append utility
# -----------------------------------------------------------------------------
strip_and_append() {
    local pattern="$1" replacement="$2"
    EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed -E "s/ ${pattern} [0-9]+//g")
    EXTRA_SERVER_ARGS+=" $replacement"
}
[[ -n "$OVERRIDE_CHECKPOINT_EVERY" ]] && strip_and_append --checkpoint-min-step "--checkpoint-min-step $OVERRIDE_CHECKPOINT_EVERY"
[[ -n "$OVERRIDE_CTX_CHECKPOINTS" ]]   && strip_and_append --ctx-checkpoints "--ctx-checkpoints $OVERRIDE_CTX_CHECKPOINTS"
[[ -n "$OVERRIDE_CACHE_RAM" ]]         && strip_and_append --cache-ram "--cache-ram $OVERRIDE_CACHE_RAM"

# -----------------------------------------------------------------------------
# Profile summary
# -----------------------------------------------------------------------------
_extract_arg() {
    local flag="$1" fallback="$2"
    local val
    val=$(echo "$EXTRA_SERVER_ARGS" | sed -nE "s/.* ${flag} ([^ ]+).*/\\1/p" | head -1)
    [[ -z "$val" ]] && val="$fallback"
    echo "$val"
}

_has_flag() {
    local flag="$1"
    if [[ "$EXTRA_SERVER_ARGS" == *" ${flag} "* || "$EXTRA_SERVER_ARGS" == *" ${flag}" ]]; then
        echo yes
    else
        echo no
    fi
}

print_profile_summary() {
    local cache_ram_val load_mode flash_attn slot_sim
    local spec_type spec_n_max spec_draft
    local moe_strategy checkpoint_min checkpoint_count no_near_end
    local chat_template

    cache_ram_val=$(_extract_arg --cache-ram "0")
    load_mode=$(_extract_arg --load-mode "${LOAD_MODE:-dio}")
    flash_attn=$(_extract_arg --flash-attn "${LLAMA_FLASH_ATTN}")
    slot_sim=$(_extract_arg --slot-prompt-similarity "")
    spec_type=$(_extract_arg --spec-type "")
    spec_n_max=$(_extract_arg --spec-draft-n-max "")
    checkpoint_min=$(_extract_arg --checkpoint-min-step "")
    checkpoint_count=$(_extract_arg --ctx-checkpoints "")

    local _ct
    _ct=$(echo "$EXTRA_SERVER_ARGS" | sed -nE "s/.*--chat-template-file '([^']+)'.*/\\1/p" | head -1)
    [[ -z "$_ct" ]] && chat_template="model default" || chat_template=$(basename "$_ct")

    spec_draft=$(echo "$EXTRA_SERVER_ARGS" | sed -nE 's/.* -md ([^ ]+).*/\1/p' | head -1)
    if [[ -n "$spec_draft" ]]; then
        spec_draft=$(basename "$spec_draft" .gguf)
    fi

    if [[ "$EXTRA_SERVER_ARGS" == *" --cpu-moe "* ]]; then
        if [[ "$EXTRA_SERVER_ARGS" == *" --load-mode none "* ]]; then
            moe_strategy="cpu-moe + --load-mode none (RAM)"
        else
            moe_strategy="cpu-moe (mmap)"
        fi
    elif [[ "$EXTRA_SERVER_ARGS" == *" --moe-expert-residency "* ]]; then
        moe_strategy="residency (madvise tracking)"
    else
        moe_strategy="GPU (--no-residency)"
    fi

    if [[ -n "$spec_type" ]]; then
        spec_type="${spec_type%%,*}"
        spec_type="draft-${spec_type#draft-}"
    fi
    no_near_end=$(_has_flag --no-checkpoint-near-end)

    printf '%b─── Profile ──────────────────────────────────────────────%b\n' "$BLUE" "$NC"
    printf '  %-28s %s\n' "Model:"           "$(basename "$MODEL" .gguf)"
    printf '  %-28s %s\n' "Profile:"         "${profile_name:-unknown}"
    printf '  %-28s %s\n' "Model size:"      "$((MODEL_BYTES / 1073741824)) GiB"
    printf '  %-28s %s\n' "Backend:"         "${BACKEND:-auto}"
    printf '  %-28s %s\n' "GPU layers:"      "${GPU_LAYERS:-99}"
    printf '  %-28s %s\n' "Context size:"    "${CTX_SIZE}"
    printf '  %-28s %s\n' "Threads (batch/gen):" "${THREADS_BATCH}/${THREADS}"
    printf '  %-28s %s/%s\n' "KV cache type:"   "${KV_CACHE_TYPE_K}" "${KV_CACHE_TYPE_V}"
    printf '  %-28s %s\n' "Batch/UBatch:"    "${OVERRIDE_BATCH_SIZE:-default}"
    printf '  %-28s %s\n' "Spec decode:"     "$( [[ -n "$spec_type" ]] && echo "$spec_type (n_max=$spec_n_max, draft=${spec_draft:-self})" || echo "off" )"
    printf '  %-28s %s\n' "MoE experts:"     "$moe_strategy"
    printf '  %-28s %s\n' "Load mode:"       "$load_mode"
    printf '  %-28s %s\n' "Flash attention:" "$flash_attn"
    printf '  %-28s %s\n' "Reasoning:"       "${OVERRIDE_REASONING:-off} (budget=${OVERRIDE_REASONING_BUDGET:-0})"
    printf '  %-28s %s\n' "Cache RAM:"       "$( [[ "$cache_ram_val" == "0" ]] && echo disabled || echo "${cache_ram_val} MiB" )"
    printf '  %-28s %s\n' "Checkpoints:"     "$( [[ -n "$checkpoint_min" ]] && echo "min=$checkpoint_min, count=${checkpoint_count:--}${no_near_end:+ (no-near-end)}" || echo "off" )"
    printf '  %-28s %s\n' "Slot similarity:" "${slot_sim:-default}"
    printf '  %-28s %s\n' "Chat template:"   "$chat_template"
    printf '  %-28s %s\n' "SSD cache:"       "$( [[ "${_SSD_DISABLE:-false}" == "true" ]] && echo disabled || echo "${SSD_PATH:-default}" )"
    printf '  %-28s %s\n' "System cache:"    "$( [[ "${_SSD_DISABLE:-false}" == "true" ]] && echo disabled || echo "${SSD_SYSTEM_PROMPTS:-off} entries" )"
    printf '  %-28s %s\n' "SSD no-fsync:"    "${SSD_NO_FSYNC:-no}"
    printf '  %-28s %s\n' "Mlock:"           "$( [[ "$(ulimit -l 2>/dev/null)" == "unlimited" ]] && echo yes || echo "no (limit: $(ulimit -l) KiB)" )"

    if [[ ${GPU_BUDGET_BYTES:-0} -gt 0 ]]; then
        local budget_gib=$(awk -v b="$GPU_BUDGET_BYTES" 'BEGIN { printf "%.1f", b / 1073741824 }')
        printf '%b─── RAM Budget ─────────────────────────────────────────────%b\n' "$BLUE" "$NC"
        printf '  %-28s %s GiB\n' "GPU budget:" "$budget_gib"
        printf '  %-28s %s GiB\n' "Target model:" "$(awk -v b="$MODEL_BYTES" 'BEGIN { printf "%.1f", b / 1073741824 }')"
        printf '  %-28s %s MiB\n' "Available for cache-ram:" "${cache_ram_val}"
        printf '%b──────────────────────────────────────────────────────────%b\n' "$BLUE" "$NC"
    fi
}

if $PRINT_PROFILE; then
    print_profile_summary
    echo ""
    cat <<PROFILE_EOF
CTX_SIZE=$CTX_SIZE
MODEL_PATH='$MODEL'
MODEL_NAME=$(basename "$MODEL" .gguf)
MODEL_BYTES=$MODEL_BYTES
GPU_LAYERS=$GPU_LAYERS
THREADS=$THREADS
THREADS_BATCH=$THREADS_BATCH
KV_CACHE_TYPE_K=$KV_CACHE_TYPE_K
KV_CACHE_TYPE_V=$KV_CACHE_TYPE_V
OVERRIDE_BATCH_SIZE='${OVERRIDE_BATCH_SIZE:-}'
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

# -----------------------------------------------------------------------------
# Final command assembly
# -----------------------------------------------------------------------------
[[ "${_SSD_DISABLE:-false}" != "true" ]] && [[ -z "$SSD_PATH" ]] && SSD_PATH="$PROJECT_ROOT/kv-cache"
setup_backend_env
apply_backend_env
setup_performance

LLAMA_SERVER=$(get_llama_binary server)
LLAMA_BIN=$(get_llama_binary cli)
[[ ! -x "$LLAMA_BIN" ]] && { log_error "Binary not found: $LLAMA_BIN"; exit 1; }
echo -e "${BLUE}Using backend: ${GREEN}${BACKEND}${NC}"

# Pre-flight memory checks
_total_mem_gb=$(($(get_total_memory_bytes) / 1073741824))
_model_size_gb=$((MODEL_BYTES / 1073741824))
if [[ $_model_size_gb -gt $((_total_mem_gb - 2)) ]]; then
    echo -e "${YELLOW}Warning: model (${_model_size_gb}GB) is close to system RAM (${_total_mem_gb}GB).${NC}"
fi

# GPU visibility check
if [[ "$BACKEND" =~ ^(vulkan|rocm)$ && ${GPU_LAYERS:-99} -ge 50 && $MODEL_BYTES -gt 0 ]]; then
    _gpu_vis_bytes=0
    for c in /sys/class/drm/card[0-9]/device; do
        [[ -d "$c" ]] || continue
        _gpu_vis_bytes=$(( $(cat "$c/mem_info_vram_total" 2>/dev/null || echo 0) + $(cat "$c/mem_info_gtt_total" 2>/dev/null || echo 0) ))
        break
    done
    _gpu_vis_gb=$((_gpu_vis_bytes / 1073741824))
    if [[ $_gpu_vis_gb -gt 0 ]]; then
        _unified_heap=0
        grep -q "radv_enable_unified_heap_on_apu" "$HOME/.drirc" /usr/share/drirc.d/*.conf 2>/dev/null && _unified_heap=1
        _gpu_budget_gb=$(( _gpu_vis_gb - 2 ))
        [[ "$BACKEND" == "vulkan" && $_unified_heap -eq 0 ]] && _gpu_budget_gb=$(( _gpu_vis_gb * 2 / 3 - 2 ))
        if [[ $MODEL_BYTES -gt $((_gpu_budget_gb * 1073741824)) ]] && [[ "$EXTRA_SERVER_ARGS" != *"--cpu-moe"* ]]; then
            log_error "Model (${_model_size_gb} GiB) exceeds GPU-visible memory budget (${_gpu_budget_gb} GiB of ${_gpu_vis_gb} GiB VRAM+GTT)."
            exit 1
        fi
        log_info "GPU memory check: model ${_model_size_gb} GiB fits ${_gpu_vis_gb} GiB VRAM+GTT (budget ${_gpu_budget_gb} GiB)"
    fi
fi

MODEL_SIZE=$(du -h "$MODEL" 2>/dev/null | cut -f1)
[[ "${MODEL_SIZE: -1}" == "M" && $MODEL_BYTES -gt 1073741824 ]] && MODEL_SIZE="$((MODEL_BYTES / 1073741824))G"
echo -e "${BLUE}Model: ${GREEN}$(basename "$MODEL" .gguf)${NC} ($MODEL_SIZE)"

if [[ "$OVERRIDE_FIT" == "on" ]]; then
    GPU_LAYERS=-1
    EXTRA_SERVER_ARGS+=" --fit on"
fi

# Common arguments
COMMON_ARGS="-m '$MODEL'"
[[ -n "$MODEL_ALIAS" ]] && COMMON_ARGS+=" -a '$MODEL_ALIAS'"
COMMON_ARGS+=" -c $CTX_SIZE --threads $THREADS --threads-batch $THREADS_BATCH"
COMMON_ARGS+=" ${OVERRIDE_BATCH_SIZE} -ngl $GPU_LAYERS"
[[ -n "$OVERRIDE_UBATCH_SIZE" ]] && COMMON_ARGS=$(echo "$COMMON_ARGS" | sed -E 's/--ubatch-size [0-9]+/--ubatch-size '"$OVERRIDE_UBATCH_SIZE"'/')
COMMON_ARGS+=" --cache-type-k $KV_CACHE_TYPE_K --cache-type-v $KV_CACHE_TYPE_V"
COMMON_ARGS+=" --load-mode $LOAD_MODE"
[[ -n "$EXTRA_COMMON_ARGS" ]] && COMMON_ARGS+=" $EXTRA_COMMON_ARGS"

# Server arguments
SERVER_ARGS="--host $HOST --port $PORT -fa on --jinja"
SERVER_ARGS+=" --reasoning ${OVERRIDE_REASONING:-off}"
[[ -n "$OVERRIDE_REASONING_BUDGET" && "$OVERRIDE_REASONING_BUDGET" != "0" ]] && SERVER_ARGS+=" --reasoning-budget $OVERRIDE_REASONING_BUDGET"
SERVER_ARGS+=" -np ${OVERRIDE_N_PARALLEL:-1} --prio ${LLAMA_PRIO} --prio-batch ${LLAMA_PRIO_BATCH} --metrics"
SERVER_ARGS+=" -ctxcp ${LLAMA_CTXCP} --cache-reuse ${LLAMA_CACHE_REUSE}"
SERVER_ARGS+=" --slot-save-path $PROJECT_ROOT/kv-cache"
SERVER_ARGS+=" --slot-prompt-similarity ${LLAMA_SLOT_PROMPT_SIMILARITY} --kv-unified"

[[ -n "$EXTRA_SERVER_ARGS" ]] && SERVER_ARGS+=" $EXTRA_SERVER_ARGS"

if [[ "${ENABLE_CONTEXT_SHIFT:-1}" == "0" ]]; then
    SERVER_ARGS+=" --no-context-shift"
    log_info "Context shifting DISABLED"
else
    log_info "Context shifting ENABLED"
fi

[[ "$PRESERVE_REASONING" == "true" ]] && SERVER_ARGS+=" --reasoning-preserve --chat-template-kwargs '{\"preserve_thinking\":true}'"

if [[ -n "$SSD_PATH" && "${_SSD_DISABLE:-false}" != "true" ]]; then
    mkdir -p "$SSD_PATH"
    SERVER_ARGS+=" --cache-ssd $SSD_PATH"
    [[ -n "$SSD_CHECKPOINTS" ]] && SERVER_ARGS+=" --cache-ssd-checkpoints $SSD_CHECKPOINTS"
    [[ -n "$SSD_HOT_WINDOW" ]] && SERVER_ARGS+=" --cache-ssd-hot-window $SSD_HOT_WINDOW"
    [[ -n "$SSD_WARM_WINDOW" ]] && SERVER_ARGS+=" --cache-ssd-warm-window $SSD_WARM_WINDOW"
    [[ -n "$SSD_MAX_COLD" ]] && SERVER_ARGS+=" --cache-ssd-max-cold $SSD_MAX_COLD"
    [[ -n "$SSD_PAGE_SIZE" ]] && SERVER_ARGS+=" --cache-ssd-page-size $SSD_PAGE_SIZE"
    [[ -n "$SSD_HOT_RAM" ]] && SERVER_ARGS+=" --cache-ssd-hot-ram $SSD_HOT_RAM"
    [[ -n "$SSD_WARM_RAM" ]] && SERVER_ARGS+=" --cache-ssd-warm-ram $SSD_WARM_RAM"
    [[ -n "${SSD_COLD_MAX_SIZE:-}" ]] && SERVER_ARGS+=" --cache-ssd-cold-maxsize $SSD_COLD_MAX_SIZE"
    [[ -n "${SSD_SYSTEM_PROMPTS:-}" ]] && SERVER_ARGS+=" --cache-ssd-system-prompts $SSD_SYSTEM_PROMPTS"
    [[ "${SSD_NO_FSYNC:-}" == "true" ]] && SERVER_ARGS+=" --cache-ssd-no-fsync"
    [[ -n "${SSD_SYSTEM_MAX_DAYS:-}" ]] && SERVER_ARGS+=" --cache-ssd-system-max-days $SSD_SYSTEM_MAX_DAYS"
    [[ -n "$PROMPT_MAX" && "$PROMPT_MAX" != "0" ]] && SERVER_ARGS+=" --prompt-max $PROMPT_MAX"
fi

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
kill_existing_server() {
    pkill -9 llama-server 2>/dev/null || true
    if command -v lsof &>/dev/null; then
        lsof -ti tcp:"$1" 2>/dev/null | xargs kill -9 2>/dev/null || true
    fi
    sleep 1
}

EXEC_ENV=""
case "$BACKEND" in
    rocm)   EXEC_ENV="LD_LIBRARY_PATH='$LD_LIBRARY_PATH'" ;;
    vulkan) EXEC_ENV="LD_LIBRARY_PATH='$LD_LIBRARY_PATH'" ;;
esac

if $SERVER_MODE; then
    kill_existing_server "$PORT"
    print_profile_summary
    echo -e "${BLUE}Starting server on ${HOST}:${PORT}...${NC}"
    echo "Command: $LLAMA_SERVER $COMMON_ARGS $SERVER_ARGS"
    eval "$EXEC_ENV" "$LLAMA_SERVER" $COMMON_ARGS $SERVER_ARGS
elif $INTERACTIVE; then
    echo "Command: $LLAMA_BIN $COMMON_ARGS -i"
    eval "$EXEC_ENV" "$LLAMA_BIN" $COMMON_ARGS -i
else
    echo "Command: $LLAMA_BIN $COMMON_ARGS -n $N_PREDICT $PROMPT"
    eval "$EXEC_ENV" "$LLAMA_BIN" $COMMON_ARGS -n $N_PREDICT "$PROMPT"
fi
