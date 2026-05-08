#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Llama.cpp Benchmark Script
# Tests models with CLIO-style 2-turn requests for prompt caching validation
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models"

# Backend paths
ROCM_BIN="$PROJECT_ROOT/src/llama-cpp-rocm/build/bin"
VULKAN_BIN="$PROJECT_ROOT/src/llama-cpp-vulkan/build/bin"

# Default settings
PORT=9090
CTX_SIZE=32768
NGL=99
MAX_TOKENS=100

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error(){ echo -e "${RED}[ERROR]${NC} $1"; }

# Models to test (format: filename:extra_flags)
MODELS=(
    "GLM-4.7-Flash-Q4_K_M.gguf:"
    "Qwen3-14B-Q5_K_M.gguf:"
    "gemma-4-26B-A4B-it-UD-Q5_K_M.gguf:"
    "Qwen3.5-27B-UD-Q5_K_XL.gguf:"
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf:"
)

# CLIO-style prompts (2 turns)
TURN1="Explain llama-run.sh"
TURN2="Explain README.md"

# =============================================================================
# Helper functions
# =============================================================================

get_binary() {
    local backend="$1"
    if [[ "$backend" == "vulkan" ]]; then
        echo "$VULKAN_BIN"
    else
        echo "$ROCM_BIN"
    fi
}

setup_backend_env() {
    local backend="$1"
    if [[ "$backend" == "rocm" ]]; then
        export ROCM_PATH="$PROJECT_ROOT/deps"
        export HIP_PATH="$ROCM_PATH"
        export HIP_PLATFORM=amd
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
        # Auto-detect GFX version
        source "$PROJECT_ROOT/scripts/detect-gpu.sh"
        export HSA_OVERRIDE_GFX_VERSION="${LLAMA_GFX_VERSION:-11.0.3}"
        # Enable unified memory for APUs (spills to GTT/system RAM via hipMallocManaged)
        export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
        unset GGML_BACKEND
    else
        export GGML_BACKEND=vulkan
        export GGML_VK_ALLOW_GRAPHICS_QUEUE=0
        export GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
        unset ROCM_PATH HIP_PATH HSA_OVERRIDE_GFX_VERSION
        # Keep existing LD_LIBRARY_PATH but remove ROCm-specific parts
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH//$PROJECT_ROOT\/deps\/lib:/}"
        else
            LD_LIBRARY_PATH=""
        fi
        export LD_LIBRARY_PATH
    fi
}

# =============================================================================
# Server Health & Error Checking
# =============================================================================

get_server_status() {
    curl -s -w "\n%{http_code}" "$API_BASE/v1/models" 2>/dev/null
}

wait_for_server() {
    local max_attempts=300
    local attempt=0
    local last_error=""
    
    while [[ $attempt -lt $max_attempts ]]; do
        local resp=$(get_server_status)
        local http_code=$(echo "$resp" | tail -1)
        local body=$(echo "$resp" | head -n -1)
        
        if [[ "$http_code" == "200" ]] && echo "$body" | grep -q '"models"'; then
            # Check if model actually loaded (look for gguf in response)
            if echo "$body" | grep -q "gguf"; then
                return 0
            fi
        fi
        
        # Check server log for errors
        if [[ -f "/tmp/llama-bench-$backend.log" ]]; then
            local error=$(grep -iE "error|failed|oom|out of memory|killed|signal" /tmp/llama-bench-$backend.log 2>/dev/null | tail -1 || echo "")
            if [[ -n "$error" ]]; then
                last_error="$error"
                # Check for fatal errors
                if echo "$error" | grep -qiE "killed|signal|segmentation|fault"; then
                    log_error "Server process died: $error"
                    return 1
                fi
            fi
        fi
        
        attempt=$((attempt + 1))
        sleep 1
    done
    
    # Report what we found
    if [[ -n "$last_error" ]]; then
        log_error "Server failed: $last_error"
    else
        log_error "Server did not respond after ${max_attempts}s"
    fi
    return 1
}

check_server_healthy() {
    local resp=$(get_server_status)
    local http_code=$(echo "$resp" | tail -1)
    [[ "$http_code" == "200" ]]
}

get_last_server_error() {
    if [[ -f "/tmp/llama-bench-$backend.log" ]]; then
        grep -iE "error|failed|oom|out of memory" /tmp/llama-bench-$backend.log 2>/dev/null | tail -3
    fi
}

start_server() {
    local model="$1"
    local extra_flags="$2"
    local backend="$3"
    
    log_info "Starting server: $model (backend: $backend)"
    
    # Kill any existing server and wait
    pkill -9 llama-server 2>/dev/null || true
    sleep 2
    
    local model_path="$MODEL_DIR/$model"
    [[ ! -f "$model_path" ]] && { log_error "Model not found: $model_path"; return 1; }
    
    # Setup backend environment
    setup_backend_env "$backend"
    
    # Get binary path
    local llama_bin=$(get_binary "$backend")
    [[ ! -x "$llama_bin/llama-server" ]] && { log_error "Binary not found: $llama_bin/llama-server"; return 1; }
    
    # Build server command - match llama-run.sh flags for consistent behavior
    local cmd=("$llama_bin/llama-server")
    cmd+=(-m "$model_path")
    cmd+=(-c "$CTX_SIZE")
    cmd+=(-ngl "$NGL")
    cmd+=(--port "$PORT")
    cmd+=(--host 0.0.0.0)
    cmd+=(--batch-size 1024 -ub 512)
    cmd+=(--cache-type-k q8_0 --cache-type-v q8_0)
    cmd+=(-fa on --jinja)
    cmd+=(--reasoning off --no-mmproj)
    cmd+=(--fit on --kv-unified)
    cmd+=(--cache-ram 4096 --cache-reuse 512)
    cmd+=(-ctxcp 64)
    
    # Add extra flags (cpu-moe, no-mmap, etc)
    [[ -n "$extra_flags" ]] && IFS=' ' read -ra FLAGS <<< "$extra_flags" && cmd+=("${FLAGS[@]}")
    
    # Start server
    "${cmd[@]}" > /tmp/llama-bench-$backend.log 2>&1 &
    local server_pid=$!
    
    log_info "Server PID: $server_pid, waiting for ready..."
    
    # Wait for server with error detection
    export API_BASE="http://localhost:$PORT"
    if ! wait_for_server; then
        log_error "Server failed to start"
        echo "=== Server log ==="
        tail -30 /tmp/llama-bench-$backend.log
        echo "=== Last errors ==="
        get_last_server_error
        return 1
    fi
    
    # Give the server a moment to settle after model load
    sleep 3
    
    log_ok "Server ready (PID: $server_pid)"
    return 0
}

benchmark_model() {
    local model="$1"
    local extra_flags="$2"
    local backend="$3"
    
    echo ""
    echo -e "${CYAN}=== Testing: $model (backend: $backend) ===${NC}"
    
    # Start server
    start_server "$model" "$extra_flags" "$backend" || return 1
    
    # Verify server is healthy before testing
    if ! check_server_healthy; then
        log_error "Server became unhealthy"
        get_last_server_error
        return 1
    fi
    
    prompt_tokens_1=0
    prompt_ms_1=0
    pred_ms_1=0
    tokens_1=0
    ttft_1=0
    tps_1=0
    
    prompt_tokens_2=0
    prompt_ms_2=0
    pred_ms_2=0
    tokens_2=0
    ttft_2=0
    tps_2=0
    
    # Turn 1 - cold (builds cache)
    echo "  Turn 1 (cache build)..."
    local resp_sync=$(curl -s -w "\n%{http_code}" "$API_BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$TURN1\"}],\"max_tokens\":$MAX_TOKENS}")
    
    local http_code=$(echo "$resp_sync" | tail -1)
    local body=$(echo "$resp_sync" | head -n -1)
    
    if [[ "$http_code" != "200" ]]; then
        log_error "API error $http_code: $body"
        return 1
    fi
    
    if echo "$body" | grep -q '"usage"'; then
        prompt_tokens_1=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null || echo "0")
        tokens_1=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
        prompt_ms_1=$(echo "$body" | python3 -c "import sys,json; print(round(json.load(sys.stdin)['timings']['prompt_ms'], 1))" 2>/dev/null || echo "0")
        pred_ms_1=$(echo "$body" | python3 -c "import sys,json; print(round(json.load(sys.stdin)['timings']['predicted_ms'], 1))" 2>/dev/null || echo "0")
        ttft_1=$(echo "$body" | python3 -c "import sys,json; print(round(json.load(sys.stdin)['timings']['predicted_ms'] / json.load(sys.stdin)['usage']['completion_tokens'], 2))" 2>/dev/null || echo "0")
        tps_1=$(python3 -c "print(round($tokens_1 * 1000 / $pred_ms_1, 2))" 2>/dev/null || echo "0")
        
        echo "    Prompt: ${prompt_ms_1}ms ($prompt_tokens_1 tokens)"
        echo "    Generation: ${pred_ms_1}ms ($tokens_1 tokens)"
        echo "    TTFT: ${ttft_1}ms/token"
        echo "    TPS: $tps_1 tok/s"
    else
        log_error "Invalid response: $body"
        return 1
    fi
    
    # Verify server still healthy
    if ! check_server_healthy; then
        log_error "Server crashed after turn 1"
        get_last_server_error
        return 1
    fi
    
    sleep 1
    
    # Turn 2 - cached (same model loads faster on second call)
    echo "  Turn 2 (cache hit)..."
    resp_sync=$(curl -s -w "\n%{http_code}" "$API_BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$TURN2\"}],\"max_tokens\":$MAX_TOKENS}")
    
    http_code=$(echo "$resp_sync" | tail -1)
    body=$(echo "$resp_sync" | head -n -1)
    
    if [[ "$http_code" != "200" ]]; then
        log_error "API error $http_code: $body"
        return 1
    fi
    
    if echo "$body" | grep -q '"usage"'; then
        prompt_tokens_2=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null || echo "0")
        tokens_2=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
        prompt_ms_2=$(echo "$body" | python3 -c "import sys,json; print(round(json.load(sys.stdin)['timings']['prompt_ms'], 1))" 2>/dev/null || echo "0")
        pred_ms_2=$(echo "$body" | python3 -c "import sys,json; print(round(json.load(sys.stdin)['timings']['predicted_ms'], 1))" 2>/dev/null || echo "0")
        ttft_2=$(echo "$body" | python3 -c "import sys,json; print(round(json.load(sys.stdin)['timings']['predicted_ms'] / json.load(sys.stdin)['usage']['completion_tokens'], 2))" 2>/dev/null || echo "0")
        tps_2=$(python3 -c "print(round($tokens_2 * 1000 / $pred_ms_2, 2))" 2>/dev/null || echo "0")
        
        echo "    Prompt: ${prompt_ms_2}ms ($prompt_tokens_2 tokens)"
        echo "    Generation: ${pred_ms_2}ms ($tokens_2 tokens)"
        echo "    TTFT: ${ttft_2}ms/token"
        echo "    TPS: $tps_2 tok/s"
    else
        log_warn "Turn 2 failed - server may have crashed"
    fi
    
    # Cleanup server
    pkill -9 llama-server 2>/dev/null || true
    sleep 2
    
    return 0
}

# =============================================================================
# Main
# =============================================================================

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark llama.cpp with different backends.

OPTIONS:
    --backend BACKEND   Backend to test: rocm, vulkan, or both (default: both)
    --port PORT         Server port (default: 9090)
    --ctx SIZE          Context size (default: 32768)
    --ngl LAYERS        GPU layers (default: 99)
    --model MODEL       Test specific model only
    --help              Show this help

EXAMPLES:
    $(basename "$0")                    # Test both rocm and vulkan
    $(basename "$0") --backend rocm     # Test ROCm only
    $(basename "$0") --backend vulkan   # Test Vulkan only

EOF
}

# Parse args
BACKEND="both"
TEST_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --ctx)
            CTX_SIZE="$2"
            shift 2
            ;;
        --ngl)
            NGL="$2"
            shift 2
            ;;
        --model)
            TEST_MODEL="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate backend
if [[ "$BACKEND" != "rocm" && "$BACKEND" != "vulkan" && "$BACKEND" != "both" ]]; then
    log_error "Invalid backend: $BACKEND"
    usage
    exit 1
fi

echo ""
echo "========================================"
echo "  Llama.cpp Benchmark"
echo "  Backend: $BACKEND"
echo "  Port: $PORT"
echo "  Context: $CTX_SIZE"
echo "  GPU Layers: $NGL"
echo "========================================"
echo ""

# Check binaries exist
if [[ "$BACKEND" == "rocm" || "$BACKEND" == "both" ]]; then
    if [[ ! -x "$ROCM_BIN/llama-server" ]]; then
        log_error "ROCm binary not found: $ROCM_BIN/llama-server"
        log_info "Run: ./scripts/rebuild.sh rocm"
        exit 1
    fi
    log_ok "ROCm binary: $ROCM_BIN/llama-server"
fi

if [[ "$BACKEND" == "vulkan" || "$BACKEND" == "both" ]]; then
    if [[ ! -x "$VULKAN_BIN/llama-server" ]]; then
        log_error "Vulkan binary not found: $VULKAN_BIN/llama-server"
        log_info "Run: ./scripts/rebuild.sh vulkan"
        exit 1
    fi
    log_ok "Vulkan binary: $VULKAN_BIN/llama-server"
fi

# Filter models if specific one requested
if [[ -n "$TEST_MODEL" ]]; then
    MODELS=("$TEST_MODEL:")
fi

# Run benchmarks
for backend in rocm vulkan; do
    if [[ "$BACKEND" == "both" ]] || [[ "$BACKEND" == "$backend" ]]; then
        echo ""
        echo -e "${GREEN}=== $backend backend ===${NC}"
        
        for model_entry in "${MODELS[@]}"; do
            IFS=':' read -r model extra_flags <<< "$model_entry"
            
            # Check model exists
            if [[ ! -f "$MODEL_DIR/$model" ]]; then
                log_warn "Skipping $model (not found)"
                continue
            fi
            
            # Run benchmark
            benchmark_model "$model" "$extra_flags" "$backend" || {
                log_error "Benchmark failed for $model on $backend"
                continue
            }
        done
    fi
done

echo ""
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo ""
