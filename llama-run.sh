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

MODEL_DIR="$PROJECT_ROOT/models"

get_backend_binary() {
    local backend="$1"
    case "$backend" in
        rocm)
            echo "$PROJECT_ROOT/src/llama-cpp-rocm/build"
            ;;
        vulkan)
            echo "$PROJECT_ROOT/src/llama-cpp-vulkan/build"
            ;;
        cpu)
            echo "$PROJECT_ROOT/src/llama-cpp-vulkan/build"
            ;;
        auto)
            # Check which is available
            if [[ -x "$PROJECT_ROOT/src/llama-cpp-rocm/build/bin/llama-server" ]]; then
                echo "$PROJECT_ROOT/src/llama-cpp-rocm/build"
            elif [[ -x "$PROJECT_ROOT/src/llama-cpp-vulkan/build/bin/llama-server" ]]; then
                echo "$PROJECT_ROOT/src/llama-cpp-vulkan/build"
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
    
    # Check for Vulkan first (default backend - best stability on RDNA3)
    if [[ -x "$PROJECT_ROOT/src/llama-cpp-vulkan/build/bin/llama-server" ]]; then
        BACKEND="vulkan"
        return 0
    fi
    
    # Check for ROCm (optional backend - known issues with some archs)
    if [[ -x "$PROJECT_ROOT/src/llama-cpp-rocm/build/bin/llama-server" ]]; then
        BACKEND="rocm"
        return 0
    fi
    
    # Fallback
    BACKEND="vulkan"
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

BACKEND="auto"
MODEL=""
MODEL_ALIAS=""
CTX_SIZE=65536
USER_CTX_SIZE=""  # set when user explicitly passes -c
N_PREDICT=256
GPU_LAYERS=99
KV_CACHE_TYPE_K="bf16"
KV_CACHE_TYPE_V="bf16"
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
PROMPT_MAX="8"
SSD_CHECKPOINTS="64"
OVERRIDE_CHECKPOINT_EVERY=""
OVERRIDE_CTX_CHECKPOINTS=""
OVERRIDE_CACHE_RAM=""
OVERRIDE_REASONING_BUDGET=""
PRESERVE_REASONING=""
OVERRIDE_N_PARALLEL=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat << USAGE
${BLUE}Llama.cpp Runner${NC} - Unified runner for Vulkan, ROCm, and CPU

${YELLOW}Usage:${NC}
    $0 [options] [model_or_file] [-- server options]
    $0 --download MODEL [--quant QUANT]

${YELLOW}Options:${NC}
    -b, --backend BACKEND    Backend: auto, rocm, vulkan, cpu
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

assign_profile() {
    local model_path="$1"
    local filename=$(basename "$model_path")
    local size_bytes=$(stat -c%s "$model_path" 2>/dev/null || echo 0)
    local size_gb=$((size_bytes / 1024 / 1024 / 1024))
    
    # Reset all variables to sensible defaults
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
    KV_CACHE_TYPE_K="bf16"
    KV_CACHE_TYPE_V="bf16"
    GPU_LAYERS=99
    EXTRA_COMMON_ARGS=""
    EXTRA_SERVER_ARGS=""
    OVERRIDE_REASONING=""
    OVERRIDE_BATCH_SIZE=""
    EXTRA_SERVER_ARGS+=" --no-mmproj"
    
    # Detect model characteristics from filename
    local is_moe=false
    local is_ssm=false
    
    if echo "$filename" | grep -qiE "moe|a3b|a8b"; then
        is_moe=true
    fi
    if echo "$filename" | grep -qiE "ssm|mamba"; then
        is_ssm=true
    fi
    
    # Profile selection based on characteristics
    if [[ "$is_ssm" == true ]]; then
        # SSM/Mamba models: cache_reuse doesn't work, need different settings
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=65536
        KV_CACHE_TYPE_K="bf16"
        KV_CACHE_TYPE_V="bf16"
        GPU_LAYERS=99
        EXTRA_SERVER_ARGS+=" --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00"
        # No checkpoint strategy - SSM models handle context internally
        # No reasoning format - SSM models don't support it
        EXTRA_SERVER_ARGS+=" --no-context-shift --checkpoint-every-n-tokens 0 --ctx-checkpoints 0 --cache-ram 6144"
        OVERRIDE_REASONING="on"
        OVERRIDE_REASONING_BUDGET="2048"
        # SSM models don't support llama_state_seq_set_data_ext, so no SSD cache
        SSD_PATH=""
        profile_name="ssm-optimized"
    elif [[ "$is_moe" == true ]]; then
        # MoE models: q8_0 KV cache for all sizes to ensure stable operation
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
        KV_CACHE_TYPE_K="bf16"
        KV_CACHE_TYPE_V="bf16"
        # Smaller batch sizes to reduce KV cache fragmentation during generation
        OVERRIDE_BATCH_SIZE="--batch-size 512 --ubatch-size 256"
        EXTRA_SERVER_ARGS+=" --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00"
        EXTRA_SERVER_ARGS+=" --repeat-penalty 1.0 --presence-penalty 0.0"
        EXTRA_SERVER_ARGS+=" --reasoning-format auto"
        # Checkpoint strategy for agentic workloads:
        # - 4096 tokens between checkpoints (fine-grained for incremental conversation growth)
        # - max 4 checkpoints per slot (balances cache hits vs memory)
        # - Each checkpoint is ~63MB, so 4 per slot * 8 conversations = ~2GB
        # - 6GB cache limit prevents over-allocation
        EXTRA_SERVER_ARGS+=" --checkpoint-every-n-tokens 4096 --ctx-checkpoints 4 --cache-ram 6144"
        # SSD cache enabled by global default
        OVERRIDE_REASONING="on"
        OVERRIDE_REASONING_BUDGET="2048"
        profile_name="moe-optimized"
    elif [[ $size_gb -gt 15 ]]; then
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
        KV_CACHE_TYPE_K="bf16"
        KV_CACHE_TYPE_V="bf16"
        OVERRIDE_BATCH_SIZE="--batch-size 512 --ubatch-size 256"
        EXTRA_SERVER_ARGS+=" --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00"
        EXTRA_SERVER_ARGS+=" --repeat-penalty 1.0 --presence-penalty 0.0"
        EXTRA_SERVER_ARGS+=" --reasoning-format auto"
        # Checkpoint strategy for agentic workloads
        EXTRA_SERVER_ARGS+=" --checkpoint-every-n-tokens 4096 --ctx-checkpoints 4 --cache-ram 6144"
        OVERRIDE_REASONING="on"
        OVERRIDE_REASONING_BUDGET="2048"
        profile_name="large-dense"
    elif [[ $size_gb -gt 10 ]]; then
        # Medium models (10-15GB): balanced settings
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=32768
        KV_CACHE_TYPE_K="bf16"
        KV_CACHE_TYPE_V="bf16"
        OVERRIDE_BATCH_SIZE="--batch-size 512 --ubatch-size 256"
        EXTRA_SERVER_ARGS+=" --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00"
        EXTRA_SERVER_ARGS+=" --repeat-penalty 1.0 --presence-penalty 0.0"
        EXTRA_SERVER_ARGS+=" --reasoning-format auto"
        EXTRA_SERVER_ARGS+=" --cache-ram 4096"
        OVERRIDE_REASONING="on"
        OVERRIDE_REASONING_BUDGET="2048"
        profile_name="medium-dense"
    else
        # Small models (<10GB): full power
        [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE=65536
        KV_CACHE_TYPE_K="bf16"
        KV_CACHE_TYPE_V="bf16"
        EXTRA_SERVER_ARGS+=" --cache-ram 4096 --slot-prompt-similarity 0.15"
        profile_name="small-efficient"
    fi
    
    echo -e "${CYAN}Auto profile: ${GREEN}$profile_name${NC} (${size_gb}GB, MoE=$is_moe, SSM=$is_ssm)"
}

# =============================================================================
# Auto-discover models from ./models directory
# =============================================================================

declare -A MODELS

scan_models() {
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo -e "${YELLOW}Warning: Models directory not found: $MODEL_DIR${NC}"
        return
    fi

    # Scan for .gguf files - use original filenames as keys
    while IFS= read -r -d '' file; do
        # Get filename without path and extension
        basename=$(basename "$file" .gguf)
        
        # Store with original basename as key
        MODELS["$basename"]="$file"
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
    for name in $(printf '%s\n' "${!MODELS[@]}" | sort); do
        model="${MODELS[$name]}"
        if [[ -f "$model" ]]; then
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
    # Set up ROCm environment for GPU detection
    export ROCM_PATH="$PROJECT_ROOT/deps"
    export LD_LIBRARY_PATH="${PROJECT_ROOT:-.}/deps/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$ROCM_PATH/bin:$PATH"
    echo -e "${BLUE}Available Backends:${NC}"
    local binary="$LLAMA_BUILD/bin/llama-cli"
    if [[ -x "$binary" ]]; then
        if [[ -n "$LLAMA_GPU_NAME" ]]; then
            echo -e "  ${GREEN}[*] ROCm/HIP${NC}   - $LLAMA_GPU_NAME ($LLAMA_GFX_ARCH)"
        else
            echo -e "  ${CYAN}[ ] ROCm/HIP${NC}   - installed (GPU not in detection map)"
        fi
    else
        echo -e "  ${YELLOW}[ ] ROCm/HIP${NC}   - not built"
    fi
    if [[ -x "$LLAMA_BUILD/bin/llama-cli" ]]; then
        echo -e "  ${GREEN}[*] Vulkan${NC}      - available"
    else
        echo -e "  ${YELLOW}[ ] Vulkan${NC}      - not built"
    fi
    echo -e "  ${GREEN}[*] CPU${NC}         - always available"
    exit 0
}

setup_performance() {
    # CPU frequency (needs sudo, graceful fallback)
    if command -v cpupower &>/dev/null; then
        sudo cpupower frequency-set -g performance 2>/dev/null || true
    fi
    
    # CPU energy performance preference (needs sudo, graceful fallback)
    for p in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do
        echo balance_performance | sudo tee "$p" 2>/dev/null || true
    done
    
    # GPU DPM (needs sudo, graceful fallback)
    if [[ -f /sys/class/drm/card0/device/power_dpm_force_performance_level ]]; then
        echo "auto" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null || true
    fi
}

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

# Get metadata for a specific file in a repo
# Returns JSON with size, etag, etc.
hf_file_info() {
    local repo="$1"
    local filename="$2"
    hf_api_get "repo_info?repo_id=$repo&repoType=model&file=$filename" 2>/dev/null
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
    
    # Extract matching GGUF files
    local matches=()
    while IFS= read -r f; do
        [[ "$f" == *.gguf ]] || continue
        # Match: Q4_K_M style OR unsloth UD-Q4_K_M style OR IQ4_NL style
        # Order matters - most specific first
        if [[ "$f" =~ [-_]${quant_pattern}[-_.] ]] || \
           [[ "$f" =~ [-_]${quant_pattern}$ ]] || \
           [[ "$f" =~ [-_]${quant_base}[-_.] ]] || \
           [[ "$f" =~ [-_]${quant_base}$ ]] || \
           [[ "$f" =~ [-_]${quant_family}[-_] ]]; then
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
    
    # Priority: exact quant match > same quant family > largest available
    while IFS= read -r f; do
        if [[ "$f" =~ [-_]${quant}[-_.] ]] || [[ "$f" =~ [-_]${quant}$ ]] || [[ "$f" == *"${quant}"*.gguf ]]; then
            selected="$f"
            break
        fi
    done <<< "$candidates"
    
    # Fallback: same quant family (e.g., Q4_K_M -> Q4_K_S)
    if [[ -z "$selected" ]]; then
        local quant_family="${quant%%_*}"
        while IFS= read -r f; do
            if [[ "$f" =~ [-_]${quant_family} ]]; then
                selected="$f"
                break
            fi
        done <<< "$candidates"
    fi
    
    # Fallback: largest file
    if [[ -z "$selected" ]]; then
        selected=$(echo "$candidates" | head -1)
    fi
    
    filename="$selected"
    
    if [[ -z "$filename" ]]; then
        echo -e "${RED}Could not determine filename${NC}"
        return 1
    fi
    
    mkdir -p "$target_dir"
    
    echo -e "\n${BLUE}Model Download Information${NC}"
    echo ""
    echo -e "  ${GREEN}Repo:${NC}     $repo"
    echo -e "  ${GREEN}File:${NC}     $filename"
    echo -e "  ${GREEN}Quant:${NC}    $quant (requested)"
    echo -e "  ${GREEN}Target:${NC}   $target_dir"
    echo ""
    
    # Check if already cached
    if [[ -f "$target_dir/$filename" ]]; then
        echo -e "${GREEN}Model already exists: $target_dir/$filename${NC}"
        return 0
    fi
    
    # Download using available tools
    local download_cmd=""
    local download_args=()
    
    if command -v hf &>/dev/null; then
        download_cmd="hf"
        download_args=(download "$repo" "$filename" --local-dir "$target_dir")
    elif command -v huggingface-cli &>/dev/null; then
        download_cmd="huggingface-cli"
        download_args=(download "$repo" "$filename" --local-dir "$target_dir" --local-dir-use-symlinks False)
    elif python3 -c "import huggingface_hub" 2>/dev/null; then
        download_cmd="python3"
        download_args=(-c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='$repo',
    filename='$filename',
    local_dir='$target_dir',
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {path}')
")
    else
        echo -e "${YELLOW}huggingface-cli not found. Install it with:${NC}"
        echo ""
        echo -e "  ${CYAN}# Option 1: pip${NC}"
        echo -e "  pip install huggingface_hub"
        echo ""
        echo -e "  ${CYAN}# Option 2: conda${NC}"
        echo -e "  conda install -c conda-forge huggingface_hub"
        echo ""
        echo -e "  ${CYAN}# Option 3: Manual download${NC}"
        echo -e "  1. Go to: https://huggingface.co/$repo"
        echo -e "  2. Download: $filename"
        echo -e "  3. Place in: $target_dir"
        echo ""
        echo -e "${BLUE}Quick download command (after installing):${NC}"
        echo ""
        echo -e "  ${CYAN}huggingface-cli download $repo $filename \\${NC}"
        echo -e "  ${CYAN}    --local-dir $target_dir \\${NC}"
        echo -e "  ${CYAN}    --local-dir-use-symlinks False${NC}"
        echo ""
        return 1
    fi
    
    echo -e "${BLUE}Downloading with $download_cmd...${NC}"
    "$download_cmd" "${download_args[@]}"
    return $?
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
            KV_CACHE_TYPE_K="$2"; KV_CACHE_TYPE_V="$2"; shift 2 ;;
        --cache-ssd) SSD_PATH="$2"; shift 2 ;;
        --cache-ssd-checkpoints) SSD_CHECKPOINTS="$2"; shift 2 ;;
        --cache-ssd-hot-window) SSD_HOT_WINDOW="$2"; shift 2 ;;
        --cache-ssd-warm-window) SSD_WARM_WINDOW="$2"; shift 2 ;;
        --cache-ssd-max-cold) SSD_MAX_COLD="$2"; shift 2 ;;
        --cache-ssd-page-size) SSD_PAGE_SIZE="$2"; shift 2 ;;
        --prompt-max) PROMPT_MAX="$2"; shift 2 ;;
        --checkpoint-every-n-tokens)
            OVERRIDE_CHECKPOINT_EVERY="$2"; shift 2 ;;
        --ctx-checkpoints)
            OVERRIDE_CTX_CHECKPOINTS="$2"; shift 2 ;;
        --cache-ram)
            OVERRIDE_CACHE_RAM="$2"; shift 2 ;;
        --np)
            OVERRIDE_N_PARALLEL="$2"; shift 2 ;;
        --preserve-reasoning) PRESERVE_REASONING="true"; shift ;;
        --no-preserve-reasoning) PRESERVE_REASONING="false"; shift ;;
        --reasoning-budget) OVERRIDE_REASONING_BUDGET="$2"; shift 2 ;;
        --no-reasoning-budget) OVERRIDE_REASONING_BUDGET="0"; shift ;;
        --interactive|-i) INTERACTIVE=true; shift ;;
        --server|-s) SERVER_MODE=true; shift ;;
        --fit) OVERRIDE_FIT="on"; shift ;;
        --print-profile) PRINT_PROFILE=true; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --list-models) list_models ;;
        --list-backends) list_backends ;;
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
elif [[ -n "$MODEL" ]] && [[ -v MODELS["$MODEL"] ]]; then
    MODEL="${MODELS["$MODEL"]}"
fi

if [[ -z "$MODEL" ]]; then
    echo -e "${YELLOW}No model specified. Use --list-models${NC}"; exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}Model not found: $MODEL${NC}"; exit 1
fi

if [[ "$BACKEND" == "auto" ]]; then
    BACKEND=$(detect_backend)
fi

# All models use dynamic profiling based on file characteristics
assign_profile "$MODEL"

# =============================================================================

if [[ "$PRINT_PROFILE" == true ]]; then
    model_name=$(basename "$MODEL" .gguf)
    model_bytes=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
    # Suppress the "Auto profile:" line that assign_profile echo'd to stdout
    cat <<PROFILE_EOF
CTX_SIZE=$CTX_SIZE
MODEL_PATH='$MODEL'
MODEL_NAME=$model_name
MODEL_BYTES=$model_bytes
GPU_LAYERS=$GPU_LAYERS
THREADS=$THREADS
KV_CACHE_TYPE_K=$KV_CACHE_TYPE_K
KV_CACHE_TYPE_V=$KV_CACHE_TYPE_V
OVERRIDE_BATCH_SIZE='${OVERRIDE_BATCH_SIZE:-"--batch-size 1024 -ub 512"}'
OVERRIDE_REASONING='${OVERRIDE_REASONING:-off}'
OVERRIDE_REASONING_BUDGET='${OVERRIDE_REASONING_BUDGET:-0}'
EXTRA_SERVER_ARGS='${EXTRA_SERVER_ARGS:-}'
PRESERVE_REASONING='${PRESERVE_REASONING:-false}'
SSD_PATH='$SSD_PATH'
SSD_CHECKPOINTS=$SSD_CHECKPOINTS
SSD_HOT_WINDOW=$SSD_HOT_WINDOW
SSD_WARM_WINDOW=$SSD_WARM_WINDOW
SSD_MAX_COLD=$SSD_MAX_COLD
SSD_PAGE_SIZE=$SSD_PAGE_SIZE
OVERRIDE_FIT='$OVERRIDE_FIT'
PROFILE_EOF
    exit 0
fi
# Setup backend
# =============================================================================

# Default SSD cache for all non-SSM models (respects user override)
[[ -z "$SSD_PATH" ]] && SSD_PATH="$PROJECT_ROOT/kv-cache"
setup_backend_env

# Get binary paths
LLAMA_BIN=$(get_llama_binary cli)
LLAMA_SERVER=$(get_llama_binary server)

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo -e "${RED}Binary not found: $LLAMA_BIN${NC}"; exit 1
fi

echo -e "${BLUE}Using backend: ${GREEN}${BACKEND}${NC}"
echo -e "${BLUE}Binary: ${GREEN}$LLAMA_SERVER${NC}"

setup_performance

MODEL_SIZE=$(du -h "$MODEL" 2>/dev/null | cut -f1)
MODEL_BYTES=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
MODEL_NAME=$(basename "$MODEL" .gguf)
echo -e "${BLUE}Model: ${GREEN}$MODEL_NAME${NC} ($MODEL_SIZE)"

# =============================================================================
# Fit mode: auto-calculate GPU layers to fit available VRAM
# =============================================================================

if [[ "$OVERRIDE_FIT" == "on" ]]; then
    GPU_LAYERS=-1
    EXTRA_SERVER_ARGS+=" --fit on"
else
    EXTRA_SERVER_ARGS+=" --fit off"
fi

# Build args
# =============================================================================

COMMON_ARGS="-m '$MODEL'"
[[ -n "$MODEL_ALIAS" ]] && COMMON_ARGS="$COMMON_ARGS -a '$MODEL_ALIAS'"
MODEL_BYTES=${MODEL_BYTES:-0}
MEMLOCK_LIMIT_KB=$(ulimit -l 2>/dev/null || echo 0)
MEMLOCK_LIMIT_BYTES=$((MEMLOCK_LIMIT_KB * 1024))
if [[ "$MODEL_BYTES" -gt 0 && "$MODEL_BYTES" -gt "$MEMLOCK_LIMIT_BYTES" ]]; then
    log_info "mlock disabled: model ($((MODEL_BYTES / 1048576)) MiB) larger than memlock limit ($((MEMLOCK_LIMIT_BYTES / 1048576)) MiB)"
else
    COMMON_ARGS="$COMMON_ARGS --mlock"
fi
COMMON_ARGS="$COMMON_ARGS -c $CTX_SIZE --threads $THREADS --threads-batch $THREADS"
COMMON_ARGS="$COMMON_ARGS ${OVERRIDE_BATCH_SIZE:---batch-size 1024 -ub 512} -ngl $GPU_LAYERS"
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

# Apply command-line overrides (strip from EXTRA_SERVER_ARGS and append fresh)
if [[ -n "$OVERRIDE_CHECKPOINT_EVERY" ]]; then
    EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed -E 's/--checkpoint-every-n-tokens [0-9]+//g')
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --checkpoint-every-n-tokens $OVERRIDE_CHECKPOINT_EVERY"
fi
if [[ -n "$OVERRIDE_CTX_CHECKPOINTS" ]]; then
    EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed -E 's/--ctx-checkpoints [0-9]+//g')
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --ctx-checkpoints $OVERRIDE_CTX_CHECKPOINTS"
fi
if [[ -n "$OVERRIDE_CACHE_RAM" ]]; then
    EXTRA_SERVER_ARGS=$(echo "$EXTRA_SERVER_ARGS" | sed -E 's/--cache-ram [0-9]+//g')
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --cache-ram $OVERRIDE_CACHE_RAM"
fi

[[ -n "$EXTRA_SERVER_ARGS" ]] && SERVER_ARGS="$SERVER_ARGS $EXTRA_SERVER_ARGS"

# Preserve reasoning/thinking in prior assistant messages
# Default: off (the agentic harness preserves knowledge, reasoning in context is redundant)
if [[ "$PRESERVE_REASONING" == "true" ]]; then
    SERVER_ARGS="$SERVER_ARGS --chat-template-kwargs '{\"preserve_thinking\":true}'"
fi



# SSD-backed KV cache
if [[ -n "$SSD_PATH" ]]; then
    mkdir -p "$SSD_PATH"
    SERVER_ARGS="$SERVER_ARGS --cache-ssd $SSD_PATH"
    [[ -n "$SSD_CHECKPOINTS" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-checkpoints $SSD_CHECKPOINTS"
    [[ -n "$SSD_HOT_WINDOW" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-hot-window $SSD_HOT_WINDOW"
    [[ -n "$SSD_WARM_WINDOW" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-warm-window $SSD_WARM_WINDOW"
    [[ -n "$SSD_MAX_COLD" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-max-cold $SSD_MAX_COLD"
    [[ -n "$SSD_PAGE_SIZE" ]] && SERVER_ARGS="$SERVER_ARGS --cache-ssd-page-size $SSD_PAGE_SIZE"
    [[ -n "$PROMPT_MAX" && "$PROMPT_MAX" != "0" ]] && SERVER_ARGS="$SERVER_ARGS --prompt-max $PROMPT_MAX"
fi

# =============================================================================
# Kill existing server (ensure only one running)
# =============================================================================

kill_existing_server() {
    local port="$1"
    # Kill any llama-server processes (including ones bound to this port)
    pkill -9 llama-server 2>/dev/null || true
    # Also kill any process holding our port
    fuser -k "${port}/tcp" 2>/dev/null || true
    sleep 1
}

# =============================================================================
# Execute (no sudo needed - runs as local user)
# =============================================================================

# Build environment inline for direct execution (no sudo)
EXEC_ENV=""
if [[ "$BACKEND" == "rocm" ]]; then
    EXEC_ENV="ROCM_PATH='$ROCM_PATH' HIP_PATH='$HIP_PATH' HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION='$HSA_OVERRIDE_GFX_VERSION' LD_LIBRARY_PATH='$LD_LIBRARY_PATH'"
elif [[ "$BACKEND" == "vulkan" ]]; then
    EXEC_ENV="LD_LIBRARY_PATH='$LD_LIBRARY_PATH'"
fi

if [[ "$SERVER_MODE" == true ]]; then
    kill_existing_server "$PORT"
    echo -e "${BLUE}Starting server on ${HOST}:${PORT}...${NC}"
    eval "$EXEC_ENV" "$LLAMA_SERVER" $COMMON_ARGS $SERVER_ARGS
else
    if [[ "$INTERACTIVE" == true ]]; then
        eval "$EXEC_ENV" "$LLAMA_BIN" $COMMON_ARGS -i
    else
        eval "$EXEC_ENV" "$LLAMA_BIN" $COMMON_ARGS -n $N_PREDICT "$PROMPT"
    fi
fi

