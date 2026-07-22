#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Llama.cpp Benchmark Script - Prompt Cache Performance
# Tests SSD prompt caching by sending identical prompts at 3 sizes
# (small ~1K, medium ~5K, large ~15K tokens) in cold (empty cache)
# and warm (SSD cache restore) states. Measures prompt eval speedup.
#
# Prompts use public domain text from Project Gutenberg (Count of Monte Cristo)
# cached in scratch/pg1184.txt. Each prompt appends a short instruction.
#
# Output: per-model summary.json + summary.md, aggregate across models.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models"
SSD_CACHE_DIR="$PROJECT_ROOT/ssd-cache"
SCRATCH_DIR="$PROJECT_ROOT/scratch"

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d-%H%M)
BENCH_DIR="$PROJECT_ROOT/benchmarks/$TIMESTAMP"
mkdir -p "$BENCH_DIR"

# Backend paths
ROCM_BIN="$PROJECT_ROOT/src/cachy-llama-rocm/build/bin"
VULKAN_BIN="$PROJECT_ROOT/src/cachy-llama-vulkan/build/bin"

# Default settings
PORT=9090
# CTX_SIZE scales with hardware tier so prompt-eval benchmarks exercise
# realistic agentic context sizes. Handheld stays at 32K (matches Ayaneo
# KB limits), halo pushes to 128K (well within Strix Halo's 96GB VRAM).
case "${LLAMA_HARDWARE_TIER:-handheld}" in
    halo)     CTX_SIZE=131072 ;;
    standard) CTX_SIZE=49152 ;;
    *)        CTX_SIZE=32768 ;;
esac
NGL=99
MAX_TOKENS=128
BENCH_TIMEOUT=900

# Prompt sizes (bytes of Gutenberg text to use as prefix)
# Approximate token mapping: 1 byte ~ 0.25 tokens for English prose
# Small:  4KB  ~ 1K tokens
# Medium: 20KB ~ 5K tokens
# Large:  60KB ~ 15K tokens
PROMPT_SIZES=(
    "small:4096"
    "medium:20480"
    "large:61440"
)
PROMPT_INSTRUCTION="Summarize this passage in one sentence."

# Source text URL
GUTENBERG_URL="http://aleph.gutenberg.org/cache/epub/1184/pg1184.txt"
GUTENBERG_CACHE="$SCRATCH_DIR/pg1184.txt"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
DIM='\033[2m'
NC='\033[0m'

log_info()  { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$1"; }
log_ok()    { printf '%b[OK]%b   %s\n' "$GREEN" "$NC" "$1"; }
log_warn()  { printf '%b[WARN]%b  %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1"; }
log_header(){ printf '%b=== %s ===%b\n' "$MAGENTA" "$1" "$NC"; }

# Models to test - auto-discovered from models/ directory.
# Excludes GGUF split-file shards (e.g. model-00002-of-00005.gguf).
# Set --model to test a specific model only.
discover_models() {
    local -a found=()
    for f in "$MODEL_DIR"/*.gguf; do
        [[ -f "$f" ]] || continue
        local name
        name=$(basename "$f")
        # Skip split-file shards (contain -NNNNN-of-NNNNN in filename)
        if [[ "$name" =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; then
            continue
        fi
        found+=("$name:")
    done
    printf '%s\n' "${found[@]}"
}

mapfile -t MODELS < <(discover_models)

# If no models found, show error
if [[ ${#MODELS[@]} -eq 0 ]]; then
    log_error "No .gguf models found in $MODEL_DIR"
    exit 1
fi

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

# =============================================================================
# Model-size helpers. Large MoE models (>=20 GB) take 10+ minutes just to
# mmap the weight file on memory-constrained systems; the curl-timeout
# scaling in run_size_test() and the server-startup wait timeout below both
# consult this so the benchmark doesn't give up on models that are valid but
# slow to load.
# =============================================================================

get_model_size_bytes() {
    local model="$1"
    local path="$MODEL_DIR/$model"
    [[ -f "$path" ]] || { echo 0; return; }
    stat -c %s "$path" 2>/dev/null || stat -f %z "$path" 2>/dev/null || echo 0
}

scale_timeout_for_model() {
    local model="$1"
    local base_timeout="$2"
    local size_bytes
    size_bytes=$(get_model_size_bytes "$model")
    local size_gb=$(( size_bytes / 1024 / 1024 / 1024 ))

    # Per-model multiplier on top of the prompt-size multiplier.
    #  20 GB  -> 2x (MoE residency models take 5-10 min to mmap on Flip)
    #  30 GB  -> 3x
    #  50 GB+ -> 4x
    local model_mult=1
    if [[ $size_gb -ge 50 ]]; then
        model_mult=4
    elif [[ $size_gb -ge 30 ]]; then
        model_mult=3
    elif [[ $size_gb -ge 20 ]]; then
        model_mult=2
    fi

    echo $(( base_timeout * model_mult ))
}

setup_backend_env() {
    local backend="$1"
    source "$PROJECT_ROOT/scripts/env.sh" "$backend"
    source "$PROJECT_ROOT/scripts/detect-gpu.sh"
    [[ "$backend" == "rocm" ]] && export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
}

# =============================================================================
# Prompt builder - extracts prefix of given byte size from cached text
# =============================================================================

build_prompt() {
    local size_bytes="$1"
    local text_file="$2"

    # Strip BOM and header, take first N bytes, add instruction
    # Use process substitution to avoid SIGPIPE on sed when head exits early
    local prefix
    prefix=$(head -c "$size_bytes" <(sed '1s/^\xEF\xBB\xBF//' "$text_file"))
    # Escape for JSON embedding
    python3 -c "
import json, sys
text = sys.stdin.read()
prompt = text + '\n\n' + '$PROMPT_INSTRUCTION'
print(json.dumps(prompt))
" <<< "$prefix"
}

# =============================================================================
# Server management
# =============================================================================

SERVER_PID=""
SERVER_LOG=""

get_server_status() {
    curl -s -w "\n%{http_code}" "$API_BASE/v1/models" 2>/dev/null
}

wait_for_server() {
    # Caller can override via $1; defaults to 900s. The actual server-startup
    # time depends on model size (mmap of a 26 GB Q5_K_XL takes ~30 min on
    # the Flip), so start_server() passes a scaled value when model is large.
    local max_attempts="${1:-900}"
    local attempt=0

    printf "    Waiting for server"

    while [[ $attempt -lt $max_attempts ]]; do
        local resp
        resp=$(get_server_status)
        local http_code
        http_code=$(echo "$resp" | tail -1)
        local body
        body=$(echo "$resp" | head -n -1)

        if [[ "$http_code" == "200" ]] && echo "$body" | grep -q "gguf"; then
            return 0
        fi

        # Check if server process is still alive - this is the only reliable
        # fatal signal. The log-based check below is a backup for cases
        # where the process is alive but stuck (e.g. deadlocked on a hang).
        # Log lines like "fit_params: ... abort" or "cache_reuse is not
        # supported" are normal warnings and must not be treated as fatal.
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_error "Server exited unexpectedly (PID $SERVER_PID)"
            tail -20 "$SERVER_LOG" 2>/dev/null || true
            return 1
        fi

        if [[ -f "$SERVER_LOG" ]]; then
            # Look for unambiguous crash markers only. The word "abort"
            # appears in legitimate fit_params warnings and must not match.
            local fatal
            fatal=$(grep -iE "SIGSEGV|SIGABRT|segfault|segmentation fault|killed \(signal|out of memory|Out of memory|Killed process" "$SERVER_LOG" 2>/dev/null | tail -1 || echo "")
            if [[ -n "$fatal" ]]; then
                log_error "Server crashed: $fatal"
                return 1
            fi
        fi

        attempt=$((attempt + 1))
        if [[ $((attempt % 5)) -eq 0 ]]; then
            printf "."
        fi
        if [[ $((attempt % 30)) -eq 0 ]]; then
            printf " %ds\n" "$attempt"
        fi
        sleep 1
    done

    printf "\n"
    log_error "Server did not respond after ${max_attempts}s"
    return 1
}

start_server() {
    local model="$1"
    local extra_flags="$2"
    local backend="$3"
    local ssd_path="$4"

    log_info "Starting server: $model (backend: $backend, ssd: ${ssd_path:-none})"

    pkill -9 llama-server 2>/dev/null || true
    sleep 3

    # Get profile from llama-run.sh
    # --print-profile prints "Auto profile: ..." to stdout (for human reading)
    # plus the exportable variable assignments. Some llama-run.sh paths emit
    # additional [INFO] lines on stdout for large-MoE handling - skip them
    # by skipping every line that doesn't look like a variable assignment.
    local profile
    profile=$(./llama-run.sh --print-profile --server --model "$MODEL_DIR/$model" --backend "$backend" | grep -E '^[A-Z_][A-Z0-9_]*=') || {
        log_error "Failed to get profile from llama-run.sh"
        return 1
    }
    eval "$profile"

    setup_backend_env "$backend"

    local llama_bin
    llama_bin=$(get_binary "$backend")
    [[ ! -x "$llama_bin/llama-server" ]] && { log_error "Binary not found: $llama_bin/llama-server"; return 1; }

    local cmd=("$llama_bin/llama-server")
    cmd+=(-m "$MODEL_PATH")
    cmd+=(-c "$CTX_SIZE")
    cmd+=(-ngl "$GPU_LAYERS")
    cmd+=(--threads "$THREADS" --threads-batch "$THREADS")
    cmd+=(--port "$PORT" --host 0.0.0.0)
    cmd+=($OVERRIDE_BATCH_SIZE)
    cmd+=(--cache-type-k "$KV_CACHE_TYPE_K" --cache-type-v "$KV_CACHE_TYPE_V")
    cmd+=(-fa on --jinja)
    cmd+=(--reasoning "$OVERRIDE_REASONING")
    cmd+=(--slot-prompt-similarity 0.20)
    cmd+=(--slot-save-path "$SSD_CACHE_DIR")
    cmd+=(--kv-unified)
    cmd+=(-np 1 --prio 3 --prio-batch 3 --metrics)
    # --ctx-checkpoints: per-slot in-memory checkpoint ring for speculative decoding.
    # Use the profile value (was hardcoded to 64, which inflated VRAM use on Halo).
    cmd+=(--ctx-checkpoints "${SSD_CHECKPOINTS:-8}" --cache-reuse 512)

    if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
        IFS=' ' read -ra PROFILE_ARGS <<< "$EXTRA_SERVER_ARGS"
        cmd+=("${PROFILE_ARGS[@]}")
    fi

    if [[ -n "$ssd_path" ]]; then
        mkdir -p "$ssd_path"
        cmd+=(--cache-ssd "$ssd_path")
        [[ -n "${SSD_CHECKPOINTS:-}" ]] && cmd+=(--cache-ssd-checkpoints "$SSD_CHECKPOINTS")
        [[ -n "${SSD_HOT_WINDOW:-}" ]] && cmd+=(--cache-ssd-hot-window "$SSD_HOT_WINDOW")
        [[ -n "${SSD_WARM_WINDOW:-}" ]] && cmd+=(--cache-ssd-warm-window "$SSD_WARM_WINDOW")
        [[ -n "${SSD_MAX_COLD:-}" ]] && cmd+=(--cache-ssd-max-cold "$SSD_MAX_COLD")
        [[ -n "${SSD_PAGE_SIZE:-}" ]] && cmd+=(--cache-ssd-page-size "$SSD_PAGE_SIZE")
        [[ -n "${SSD_HOT_RAM:-}" ]] && cmd+=(--cache-ssd-hot-ram "$SSD_HOT_RAM")
        [[ -n "${SSD_WARM_RAM:-}" ]] && cmd+=(--cache-ssd-warm-ram "$SSD_WARM_RAM")
    fi

    [[ -n "$extra_flags" ]] && IFS=' ' read -ra FLAGS <<< "$extra_flags" && cmd+=("${FLAGS[@]}")

    {
        printf "BENCHMARK COMMAND:"
        for arg in "${cmd[@]}"; do
            printf " %q" "$arg"
        done
        printf "\n"
    } > "$SERVER_LOG.cmd"

    # Also dump into the server log itself - the server's > redirect below
    # truncates it, so we keep a sidecar for the parser to find.
    printf "BENCHMARK COMMAND:" >> "$SERVER_LOG"
    for arg in "${cmd[@]}"; do
        printf " %q" "$arg" >> "$SERVER_LOG"
    done
    printf "\n" >> "$SERVER_LOG"

    "${cmd[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    export API_BASE="http://localhost:$PORT"
    # Scale the server-startup wait by model size: a 26 GB Q5_K_XL takes
    # ~30 min to mmap on Flip with --moe-expert-residency, vs ~3 min for a
    # 12 GB gpt-oss. Without this, the benchmark aborts before the server
    # finishes loading the larger MoE models.
    local model_bytes
    model_bytes=$(get_model_size_bytes "$model")
    local wait_timeout=900
    if [[ $model_bytes -ge $((30 * 1024 * 1024 * 1024)) ]]; then
        wait_timeout=2700
    elif [[ $model_bytes -ge $((20 * 1024 * 1024 * 1024)) ]]; then
        wait_timeout=1800
    fi

    if ! wait_for_server "$wait_timeout"; then
        log_error "Server failed to start"
        tail -30 "$SERVER_LOG"
        return 1
    fi

    sleep 2
    log_ok "Server ready (PID: $SERVER_PID)"
    return 0
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -9 llama-server 2>/dev/null || true
    sleep 2
}

# =============================================================================
# Direct API call - send prompt, extract timing from response
# =============================================================================

call_api() {
    local prompt_json="$1"
    local run_label="$2"
    local out_dir="$3"
    local timeout="${4:-$BENCH_TIMEOUT}"

    local raw_resp_file="$out_dir/${run_label}-response.json"
    local stats_file="$out_dir/${run_label}-stats.json"

    local start_ns
    start_ns=$(date +%s%N)

    # Write response body directly to file via -o, capture HTTP code from -w
    local http_code
    http_code=$(curl -s --max-time "$timeout" \
        -w "%{http_code}" \
        -o "$raw_resp_file" \
        "$API_BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$prompt_json}],\"max_tokens\":$MAX_TOKENS}") || {
        echo "{\"error\": \"curl failed (timeout ${timeout}s)\", \"wall_ms\": 0}" > "$stats_file"
        cat "$stats_file"
        return 1
    }

    local end_ns
    end_ns=$(date +%s%N)
    local wall_ms=$(( (end_ns - start_ns) / 1000000 ))

    if [[ "$http_code" != "200" ]]; then
        local body_preview
        body_preview=$(head -c 500 "$raw_resp_file" 2>/dev/null || echo "(empty)")
        python3 -c "import json,sys
print(json.dumps({'error': 'HTTP $http_code', 'wall_ms': $wall_ms, 'body_preview': sys.stdin.read()}))" <<< "$body_preview" > "$stats_file"
        cat "$stats_file"
        return 1
    fi

    # Extract timing stats from response file (avoid shell variable JSON embedding)
    python3 - "$raw_resp_file" "$wall_ms" "$stats_file" << 'PYEOF'
import json, sys

resp_file = sys.argv[1]
wall_ms = int(sys.argv[2])
stats_file = sys.argv[3]

try:
    with open(resp_file, 'r') as f:
        data = json.load(f)

    u = data.get('usage', {})
    t = data.get('timings', {})
    pt = u.get('prompt_tokens', 0)
    ct = u.get('completion_tokens', 0)
    pms = t.get('prompt_ms', 0)
    gms = t.get('predicted_ms', 0)
    cached = u.get('prompt_tokens_details', {}).get('cached_tokens', 0)

    result = {
        'prompt_tokens': pt,
        'completion_tokens': ct,
        'cached_tokens': cached,
        'prompt_ms': round(pms, 1),
        'generation_ms': round(gms, 1),
        'total_ms': round(t.get('total_ms', pms + gms), 1),
        'ttft_ms': round(pms, 1),
        'wall_ttft_ms': wall_ms,
        'tps': round(ct * 1000 / gms, 1) if gms > 0 else 0,
        'prompt_tps': round(pt * 1000 / pms, 1) if pms > 0 else 0,
        'prompt_per_token_ms': round(t.get('prompt_per_token_ms', pms / pt if pt > 0 else 0), 1),
        'predicted_per_token_ms': round(t.get('predicted_per_token_ms', gms / ct if ct > 0 else 0), 1),
        'wall_ms': wall_ms,
    }
except Exception as e:
    result = {'error': str(e), 'wall_ms': wall_ms}

with open(stats_file, 'w') as f:
    json.dump(result, f, indent=2)
PYEOF

    cat "$stats_file"
}

# =============================================================================
# Cache hit detection from server log
# =============================================================================

detect_cache_state() {
    local log_file="$1"

    python3 -c "
import re, json

with open('$log_file', 'r') as f:
    text = f.read()

# Cache restore events. The server emits several distinct messages:
#   - 'cold-start: system prompt cache hit (n_past=N)' - the global system
#     prompt cache (cross-conversation) restored N tokens before the eval.
#   - 'loaded N checkpoints from <path>' - the SSD-backed per-conversation
#     checkpoint index was loaded on startup. N must be > 0 to count.
#   - 'restored system prompt from cache (n_sys=N, ..., skipping N tokens)'
#     - the slot actually used the restored state.
# Any of these means we have a cache hit. The legacy 'cold-start restored
# SSD' / 'restored in-memory checkpoint' strings were never emitted by
# this server, so the prior detector always reported 'miss' on warm runs.
sys_cache_hit = bool(re.search(r'system prompt cache hit', text))
sys_prompt_restored = bool(re.search(r'restored system prompt from cache', text))
loaded_match = re.search(r'loaded ([0-9]+) checkpoints from', text)
loaded_checkpoints = loaded_match is not None and int(loaded_match.group(1)) > 0
cold_restored = bool(re.search(r'cold-start restored SSD', text))
warm_restored = bool(re.search(r'restored in-memory checkpoint', text))

stored = len(re.findall(r'SSD cache: stored checkpoint', text))
divergences = len(re.findall(r'cache prefix divergence', text))

# Determine cache state
if cold_restored:
    state = 'ssd_cold'
elif warm_restored or sys_cache_hit or sys_prompt_restored or loaded_checkpoints:
    # 'ssd_warm' subsumes both in-memory and SSD-backed restores - the
    # benchmark cares about whether tokens were skipped, not the tier.
    state = 'ssd_warm'
else:
    state = 'miss'

total_checkpoints = len(re.findall(r'SSD cache: (stored|promoted|demoted) checkpoint', text))

print(json.dumps({
    'cache_state': state,
    'ssd_stored': stored,
    'divergences': divergences,
    'total_checkpoints': total_checkpoints
}))
"
}

# =============================================================================
# MoE expert residency stats. The server emits periodic warnings like:
#   W moe-residency: decodes=16 touches=1372 hits=3504 misses=636 evictions=4 hit_rate=84.6%
# We capture the last one per run (final stats) plus the activation info.
# Empty result when residency is disabled or no MoE layers.
# =============================================================================

detect_moe_residency_stats() {
    local log_file="$1"
    python3 -c "
import re, json

with open('$log_file', 'r') as f:
    text = f.read()

enabled = bool(re.search(r'moe-residency: enabled for', text))

# Find ALL hit_rate lines and take the last one (most recent decodes).
stats_lines = re.findall(
    r'moe-residency:\s+decodes=(\d+)\s+touches=(\d+)\s+hits=(\d+)\s+misses=(\d+)\s+evictions=(\d+)\s+hit_rate=([\d.]+)%',
    text
)

result = {'enabled': enabled, 'present': bool(stats_lines)}
if stats_lines:
    decodes, touches, hits, misses, evictions, hit_rate = stats_lines[-1]
    result.update({
        'decodes': int(decodes),
        'touches': int(touches),
        'hits': int(hits),
        'misses': int(misses),
        'evictions': int(evictions),
        'hit_rate_pct': float(hit_rate),
    })

# Activation info from the 'enabled for' line.
act_match = re.search(
    r'moe-residency:\s+enabled for\s+(\d+)\s+MoE layers\s+\((\d+)\s+experts,\s+(\d+)\s+used/token\)',
    text
)
if act_match:
    result['moe_layers'] = int(act_match.group(1))
    result['n_expert'] = int(act_match.group(2))
    result['n_expert_used'] = int(act_match.group(3))

print(json.dumps(result))
"
}

# =============================================================================
# Profile detection from the persisted benchmark command sidecar. We look for
# distinctive flags set by llama-run.sh's profile logic:
#   --moe-expert-residency      -> 'moe-optimized'
#   --no-checkpoint-near-end    -> 'standard' or larger
#   --cache-ram 16384           -> 'halo' tier
#   --cache-ram 4096            -> 'handheld' or 'standard'
# Plus ngl / batch-size heuristics.
# =============================================================================

detect_profile() {
    local cmd_file="$1"
    python3 -c "
import re, json, os

if not os.path.exists('$cmd_file'):
    print(json.dumps({'profile': 'unknown'}))
    exit()

with open('$cmd_file', 'r') as f:
    text = f.read()

profile = 'unknown'
features = []

if '--moe-expert-residency' in text:
    profile = 'moe-optimized'
    features.append('moe-residency')
elif re.search(r'--ctx-checkpoints 64', text):
    profile = 'dense-large'
elif re.search(r'--cache-ram 16384', text):
    profile = 'halo-tier'
elif re.search(r'--cache-ram 4096', text):
    profile = 'standard'
else:
    profile = 'default'

# Detect other notable features
if '--no-checkpoint-near-end' in text:
    features.append('no-checkpoint-near-end')
if '--kv-unified' in text:
    features.append('kv-unified')
# --cpu-moe pins MoE expert weights to host RAM (combined with
# --moe-expert-residency, enables running models that exceed the iGPU's
# VRAM+GTT budget - e.g. Q5_K_XL/Q8_K_XL on Flip's 24 GB iGPU budget).
if '--cpu-moe' in text or '--cmoe' in text:
    features.append('cpu-moe')
if 'cache-ssd-cold-maxsize' in text:
    m = re.search(r'--cache-ssd-cold-maxsize\s+(\d+)', text)
    if m:
        features.append(f'ssd-cold-maxsize={m.group(1)}MiB')

print(json.dumps({'profile': profile, 'features': features}))
"
}

# Measure SSD cache directory footprint in MiB. Helps track capacity growth
# across runs and verify cap behavior.
# =============================================================================

measure_ssd_cache() {
    local path="$1"
    python3 -c "
import os, json
total = 0
file_count = 0
try:
    for root, dirs, files in os.walk('$path'):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
                file_count += 1
            except OSError:
                pass
except OSError:
    pass
print(json.dumps({'bytes': total, 'mib': round(total / 1024 / 1024, 1), 'files': file_count}))
"
}

# =============================================================================
# Run a single test: cold + warm for one prompt size
# =============================================================================

run_size_test() {
    local model="$1"
    local extra_flags="$2"
    local backend="$3"
    local out_dir="$4"
    local size_label="$5"
    local size_bytes="$6"

    printf "    ${YELLOW}▶ %s (%d bytes)${NC}\n" "$size_label" "$size_bytes" >&2

    local prompt_json
    prompt_json=$(build_prompt "$size_bytes" "$GUTENBERG_CACHE")

    local cold_pass=true warm_pass=true

    # Scale timeout by prompt size (large prompts need more eval time) and
    # by model size (large MoE models are slower to evaluate due to
    # sparse expert routing and madvise-based residency paging).
    local timeout=$BENCH_TIMEOUT
    if [[ "$size_bytes" -ge 40000 ]]; then
        timeout=$((BENCH_TIMEOUT * 3))
    elif [[ "$size_bytes" -ge 15000 ]]; then
        timeout=$((BENCH_TIMEOUT * 2))
    fi
    timeout=$(scale_timeout_for_model "$model" "$timeout")

    # ── Cold run: empty SSD cache ───────────────────────────────────────
    rm -rf "$SSD_CACHE_DIR"
    SERVER_LOG="$out_dir/server-${size_label}-cold.log"
    start_server "$model" "$extra_flags" "$backend" "$SSD_CACHE_DIR" || return 1

    printf "      cold: " >&2
    call_api "$prompt_json" "${size_label}-cold" "$out_dir" "$timeout" > /dev/null || cold_pass=false
    local cold_stats_file="$out_dir/${size_label}-cold-stats.json"
    if $cold_pass && [[ -f "$cold_stats_file" ]]; then
        local cold_pt cold_ttft
        cold_pt=$(python3 -c "import json; print(json.load(open('$cold_stats_file')).get('prompt_tokens', 0))" 2>/dev/null || echo 0)
        cold_ttft=$(python3 -c "import json; print(json.load(open('$cold_stats_file')).get('ttft_ms', 0))" 2>/dev/null || echo 0)
        printf "${GREEN}%s tokens, TTFT %sms${NC}\n" "$cold_pt" "$cold_ttft" >&2
    else
        printf "${RED}failed${NC}\n" >&2
    fi

    local cache_cold
    cache_cold=$(detect_cache_state "$SERVER_LOG") || cache_cold='{"cache_state":"unknown"}'
    local moe_cold
    moe_cold=$(detect_moe_residency_stats "$SERVER_LOG") || moe_cold='{"enabled":false}'
    local profile
    profile=$(detect_profile "$SERVER_LOG.cmd") || profile='{"profile":"unknown"}'
    stop_server

    # ── Warm run: restart with SSD cache ─────────────────────────────────
    SERVER_LOG="$out_dir/server-${size_label}-warm.log"
    start_server "$model" "$extra_flags" "$backend" "$SSD_CACHE_DIR" || return 1

    printf "      warm: " >&2
    call_api "$prompt_json" "${size_label}-warm" "$out_dir" "$timeout" > /dev/null || warm_pass=false
    local warm_stats_file="$out_dir/${size_label}-warm-stats.json"
    if $warm_pass && [[ -f "$warm_stats_file" ]]; then
        local warm_pt warm_ttft
        warm_pt=$(python3 -c "import json; print(json.load(open('$warm_stats_file')).get('prompt_tokens', 0))" 2>/dev/null || echo 0)
        warm_ttft=$(python3 -c "import json; print(json.load(open('$warm_stats_file')).get('ttft_ms', 0))" 2>/dev/null || echo 0)
        printf "${GREEN}%s tokens, TTFT %sms${NC}\n" "$warm_pt" "$warm_ttft" >&2
    else
        printf "${RED}failed${NC}\n" >&2
    fi

    local cache_warm
    cache_warm=$(detect_cache_state "$SERVER_LOG") || cache_warm='{"cache_state":"unknown"}'
    local moe_warm
    moe_warm=$(detect_moe_residency_stats "$SERVER_LOG") || moe_warm='{"enabled":false}'
    stop_server

    # Measure SSD cache footprint after both runs.
    local ssd_size
    ssd_size=$(measure_ssd_cache "$SSD_CACHE_DIR") || ssd_size='{"mib":0,"files":0}'

    # ── Assemble result via temp file (no shell JSON embedding) ───────────
    local result_file="$out_dir/${size_label}-result.json"
    python3 - "$out_dir" "$size_label" "$size_bytes" "$cold_pass" "$warm_pass" "$cache_cold" "$cache_warm" "$moe_cold" "$moe_warm" "$profile" "$ssd_size" "$result_file" << 'PYEOF'
import json, sys, os

out_dir = sys.argv[1]
size_label = sys.argv[2]
size_bytes = int(sys.argv[3])
cold_pass = sys.argv[4] == 'true'
warm_pass = sys.argv[5] == 'true'
cache_cold = json.loads(sys.argv[6])
cache_warm = json.loads(sys.argv[7])
moe_cold = json.loads(sys.argv[8])
moe_warm = json.loads(sys.argv[9])
profile = json.loads(sys.argv[10])
ssd_size = json.loads(sys.argv[11])
result_file = sys.argv[12]

def load_stats(label):
    path = os.path.join(out_dir, f'{label}-stats.json')
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e)}

cold = load_stats(f'{size_label}-cold') if cold_pass else {'error': 'failed'}
warm = load_stats(f'{size_label}-warm') if warm_pass else {'error': 'failed'}

cold_prompt_ms = cold.get('prompt_ms', 0)
warm_prompt_ms = warm.get('prompt_ms', 0)

speedup = round(cold_prompt_ms / warm_prompt_ms, 2) if warm_prompt_ms > 0 and cold_prompt_ms > 0 else 0

# TTFT speedup (wall-clock)
cold_wall = cold.get('wall_ttft_ms', 0)
warm_wall = warm.get('wall_ttft_ms', 0)
ttft_speedup = round(cold_wall / warm_wall, 2) if warm_wall > 0 and cold_wall > 0 else 0

result = {
    'size_label': size_label,
    'size_bytes': size_bytes,
    'cold': cold,
    'warm': warm,
    'cache_cold': cache_cold,
    'cache_warm': cache_warm,
    'moe_cold': moe_cold,
    'moe_warm': moe_warm,
    'profile': profile,
    'ssd_size': ssd_size,
    'prompt_eval_speedup': speedup,
    'ttft_speedup': ttft_speedup,
    'cold_prompt_tps': cold.get('prompt_tps', 0),
    'warm_prompt_tps': warm.get('prompt_tps', 0),
    'cold_ppt_ms': cold.get('prompt_per_token_ms', 0),
    'warm_ppt_ms': warm.get('prompt_per_token_ms', 0),
    'cold_gen_ppt_ms': cold.get('predicted_per_token_ms', 0),
    'warm_gen_ppt_ms': warm.get('predicted_per_token_ms', 0),
    'cold_ttft_ms': cold.get('ttft_ms', 0),
    'warm_ttft_ms': warm.get('ttft_ms', 0),
    'cold_wall_ttft_ms': cold.get('wall_ttft_ms', 0),
    'warm_wall_ttft_ms': warm.get('wall_ttft_ms', 0),
}

with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
PYEOF
}

# =============================================================================
# Per-model benchmark
# =============================================================================

run_model_benchmark() {
    local model="$1"
    local extra_flags="$2"
    local backend="$3"

    exec 3>&1 1>&2

    local model_name
    model_name=$(basename "$model" .gguf)
    local out_dir="$BENCH_DIR/$backend/$model_name"
    mkdir -p "$out_dir"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $model_name"
    echo "  Backend: $backend"
    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local model_start=$SECONDS

    for size_entry in "${PROMPT_SIZES[@]}"; do
        IFS=':' read -r size_label size_bytes <<< "$size_entry"

        run_size_test "$model" "$extra_flags" "$backend" "$out_dir" "$size_label" "$size_bytes" || {
            log_error "Size test $size_label failed for $model_name"
            # Write an error result file so the summary can pick it up
            python3 -c "import json; json.dump({'size_label':'$size_label','error':'test failed'}, open('$out_dir/${size_label}-result.json','w'))"
            continue
        }

        # Show inline result from the result file (not shell variables)
        local result_file="$out_dir/${size_label}-result.json"
        if [[ -f "$result_file" ]]; then
            local speedup warm_state cold_ppt warm_ppt cold_ttft warm_ttft moe_hit residency_str
            speedup=$(python3 -c "import json; print(json.load(open('$result_file')).get('prompt_eval_speedup', 0))" 2>/dev/null || echo 0)
            warm_state=$(python3 -c "import json; print(json.load(open('$result_file')).get('cache_warm',{}).get('cache_state','?'))" 2>/dev/null || echo "?")
            cold_ppt=$(python3 -c "import json; print(json.load(open('$result_file')).get('cold_ppt_ms', 0))" 2>/dev/null || echo 0)
            warm_ppt=$(python3 -c "import json; print(json.load(open('$result_file')).get('warm_ppt_ms', 0))" 2>/dev/null || echo 0)
            cold_ttft=$(python3 -c "import json; print(json.load(open('$result_file')).get('cold_ttft_ms', 0))" 2>/dev/null || echo 0)
            warm_ttft=$(python3 -c "import json; print(json.load(open('$result_file')).get('warm_ttft_ms', 0))" 2>/dev/null || echo 0)
            moe_hit=$(python3 -c "import json; d=json.load(open('$result_file')).get('moe_warm',{}); print(f\"{d.get('hit_rate_pct',0):.1f}%\" if d.get('present') else '-')" 2>/dev/null || echo "-")
            residency_str=""
            if [[ "$moe_hit" != "-" ]]; then
                residency_str=" residency=${YELLOW}${moe_hit}${NC}"
            fi
            printf "    -> ${GREEN}%s${NC}: warm=${CYAN}%s${NC} speedup=${MAGENTA}%.1fx${NC} TTFT=${DIM}%s/%sms${NC} eval=${DIM}%.1f/%.1f ms/tok${NC}%s\n" \
                "$size_label" "$warm_state" "$speedup" "$cold_ttft" "$warm_ttft" "$cold_ppt" "$warm_ppt" "$residency_str"
        fi
    done

    # ── Generate per-model summary from result files ─────────────────────
    python3 - "$out_dir" "$model_name" "$backend" "$CTX_SIZE" "$MAX_TOKENS" "$TIMESTAMP" << 'PYEOF'
import json, sys, os, glob

out_dir = sys.argv[1]
model_name = sys.argv[2]
backend = sys.argv[3]
ctx_size = int(sys.argv[4])
max_tokens = int(sys.argv[5])
timestamp = sys.argv[6]

# Collect all result files
results = []
for rf in sorted(glob.glob(os.path.join(out_dir, '*-result.json'))):
    try:
        with open(rf) as f:
            results.append(json.load(f))
    except Exception as e:
        results.append({'error': str(e), 'size_label': os.path.basename(rf)})

summary = {
    'model': model_name,
    'backend': backend,
    'context': ctx_size,
    'max_tokens': max_tokens,
    'timestamp': timestamp,
    'results': results
}

with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Generate summary.md
profile_name = ''
profile_features = []
if results:
    p = results[0].get('profile', {})
    profile_name = p.get('profile', '')
    profile_features = p.get('features', [])
ssd_total_mib = 0
if results:
    ssd_total_mib = results[-1].get('ssd_size', {}).get('mib', 0)

profile_line = f' | **Profile:** {profile_name}'
if profile_features:
    profile_line += f' (features: {", ".join(profile_features)})'

md = f'''# {model_name} ({backend})

**Context:** {ctx_size} | **Output tokens/req:** {max_tokens}{profile_line}
**SSD cache footprint after run:** {ssd_total_mib} MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
'''

for r in results:
    if r.get('error'):
        md += f'| {r.get("size_label", "?")} | - | - | - | - | - | - | - | - | error | - |\n'
        continue
    c = r.get('cold', {})
    w = r.get('warm', {})
    cw = r.get('cache_warm', {})
    moe = r.get('moe_warm', {})
    moe_hit = f"{moe.get('hit_rate_pct', 0):.1f}%" if moe.get('present') else '-'
    md += f'| {r["size_label"]} | {c.get("prompt_tokens", 0)} | {r.get("cold_ttft_ms", 0)}ms | {w.get("prompt_tokens", 0)} | {r.get("warm_ttft_ms", 0)}ms | {r.get("ttft_speedup", 0)}x | {r.get("cold_ppt_ms", 0)}ms | {r.get("warm_ppt_ms", 0)}ms | {r.get("cold_gen_ppt_ms", 0)}ms | {cw.get("cache_state", "?")} | {moe_hit} |\n'

md += '''
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
'''

with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
    f.write(md)
PYEOF

    local elapsed=$(( SECONDS - model_start ))
    log_ok "$model_name complete in ${elapsed}s"
    echo "  Output: $out_dir/"

    echo "$out_dir/summary.json" >&3
}
# =============================================================================

generate_aggregate() {
    local backend="$1"
    shift
    local summary_files=("$@")

    local agg_file="$BENCH_DIR/$backend/summary.json"
    local agg_md="$BENCH_DIR/$backend/summary.md"

    # Build aggregate JSON
    python3 -c "
import json, sys

models = []
for f in sys.argv[1:]:
    try:
        with open(f, 'r') as fh:
            models.append(json.load(fh))
    except:
        pass

with open('$agg_file', 'w') as f:
    json.dump({'backend': '$backend', 'timestamp': '$TIMESTAMP', 'models': models}, f, indent=2)
" "${summary_files[@]}"

    # Build aggregate markdown
    python3 - "$agg_file" "$agg_md" << 'PYEOF'
import json, sys

agg_file = sys.argv[1]
agg_md = sys.argv[2]

with open(agg_file, 'r') as f:
    data = json.load(f)

ctx_size = data.get('context', '?')

md = f'''# Benchmark Results: {data['backend'].upper()}

**Date:** {data['timestamp']} | **Context:** {ctx_size}

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
'''

for m in data['models']:
    row = [m['model']]
    for label in ['small', 'medium', 'large']:
        found = [r for r in m['results'] if r.get('size_label') == label]
        if found:
            ttft_speedup = found[0].get('ttft_speedup', 0)
            eval_speedup = found[0].get('prompt_eval_speedup', 0)
            state = found[0].get('cache_warm', {}).get('cache_state', '?')
            cold_ttft = found[0].get('cold_ttft_ms', 0)
            warm_ttft = found[0].get('warm_ttft_ms', 0)
            row.append(f'TTFT {ttft_speedup}x ({cold_ttft}/{warm_ttft}ms, {state})')
        else:
            row.append('-')
    md += '| ' + ' | '.join(row) + ' |\n'

md += '''
**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## MoE Expert Residency (warm run)

Hit rate is the percent of expert lookups served from the in-RAM madvise
cache (vs falling through to SSD/weights). Only meaningful for MoE models
with `--moe-expert-residency` enabled; '-' otherwise.

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
'''

for m in data['models']:
    row = [m['model']]
    for label in ['small', 'medium', 'large']:
        found = [r for r in m['results'] if r.get('size_label') == label]
        if found:
            moe = found[0].get('moe_warm', {})
            if moe.get('present'):
                hit = moe.get('hit_rate_pct', 0)
                hits = moe.get('hits', 0)
                misses = moe.get('misses', 0)
                row.append(f'{hit:.1f}% ({hits}h/{misses}m)')
            else:
                row.append('-')
        else:
            row.append('-')
    md += '| ' + ' | '.join(row) + ' |\n'

md += '\n## Per-Model Detail\n'

for m in data['models']:
    profile_name = ''
    if m['results']:
        profile_name = m['results'][0].get('profile', {}).get('profile', '')
    md += f'''
### {m['model']} (profile: {profile_name})

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
'''
    for r in m['results']:
        if r.get('error'):
            md += f'| {r.get("size_label", "?")} | - | - | - | - | - | - | error | - |\n'
            continue
        cw = r.get('cache_warm', {})
        moe = r.get('moe_warm', {})
        moe_hit = f"{moe.get('hit_rate_pct', 0):.1f}%" if moe.get('present') else '-'
        md += f'| {r["size_label"]} | {r.get("cold_ttft_ms", 0)}ms | {r.get("warm_ttft_ms", 0)}ms | {r.get("ttft_speedup", 0)}x | {r.get("cold_ppt_ms", 0)}ms | {r.get("warm_ppt_ms", 0)}ms | {r.get("cold_gen_ppt_ms", 0)}ms | {cw.get("cache_state", "?")} | {moe_hit} |\n'

with open(agg_md, 'w') as f:
    f.write(md)

print(f'Aggregate written to {agg_md}')
PYEOF
}

# =============================================================================
# Download source text
# =============================================================================

fetch_source_text() {
    mkdir -p "$SCRATCH_DIR"

    if [[ -f "$GUTENBERG_CACHE" ]]; then
        local size
        size=$(wc -c < "$GUTENBERG_CACHE" 2>/dev/null || echo 0)
        if [[ "$size" -gt 100000 ]]; then
            log_ok "Source text cached: $GUTENBERG_CACHE ($size bytes)"
            return 0
        fi
    fi

    log_info "Downloading source text from Project Gutenberg..."
    if curl -sL --max-time 120 -o "$GUTENBERG_CACHE" "$GUTENBERG_URL"; then
        local size
        size=$(wc -c < "$GUTENBERG_CACHE" 2>/dev/null || echo 0)
        log_ok "Downloaded: $size bytes"
    else
        log_error "Failed to download source text"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Tests SSD prompt caching performance using direct API calls at 3 prompt sizes
(small ~1K, medium ~5K, large ~15K tokens) in cold and warm cache states.

Prompts use public domain text from The Count of Monte Cristo (Project Gutenberg).

OPTIONS:
    --backend BACKEND   Backend: rocm, vulkan, or both (default: vulkan)
    --port PORT         Server port (default: 9090)
    --ctx SIZE          Context size (default: 32768)
    --ngl LAYERS        GPU layers (default: 99)
    --tokens N          Max output tokens per request (default: 128)
    --model MODEL       Test specific model only
    --help              Show this help

OUTPUT:
    benchmarks/YYYYMMDD-HHMM/
    ├── vulkan/
    │   ├── ModelName/
    │   │   ├── server-{size}-{cold,warm}.log    # Server logs
    │   │   ├── {size}-{cold,warm}-response.json  # API responses
    │   │   ├── summary.json                     # Machine-readable
    │   │   └── summary.md                       # Human-readable
    │   └── summary.json / summary.md            # Aggregate
    └── rocm/ ...

EXAMPLES:
    $(basename "$0")                                    # Test all models on vulkan
    $(basename "$0") --backend vulkan --model GLM-4.7-Flash-Q4_K_M.gguf
    $(basename "$0") --backend both --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

EOF
}

# Parse args
BACKEND="vulkan"
TEST_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --ctx)
            CTX_SIZE="$2"; shift 2 ;;
        --ngl)
            NGL="$2"; shift 2 ;;
        --tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        --model)
            TEST_MODEL="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            log_error "Unknown option: $1"
            usage; exit 1 ;;
    esac
done

[[ "$BACKEND" != "rocm" && "$BACKEND" != "vulkan" && "$BACKEND" != "both" ]] && { log_error "Invalid backend: $BACKEND"; usage; exit 1; }

echo ""
echo "========================================"
echo "  Prompt Cache Benchmark"
echo "  Backend: $BACKEND"
echo "  Context: $CTX_SIZE"
echo "  Sizes: ${#PROMPT_SIZES[@]} (small, medium, large)"
echo "  Source: The Count of Monte Cristo (Gutenberg)"
echo "  Output: $BENCH_DIR"
echo "========================================"
echo ""

# Ensure source text is available
fetch_source_text || exit 1

# Check binaries
for be in rocm vulkan; do
    if [[ "$BACKEND" == "both" ]] || [[ "$BACKEND" == "$be" ]]; then
        bin_dir=$(get_binary "$be")
        if [[ ! -x "$bin_dir/llama-server" ]]; then
            log_error "$be binary not found: $bin_dir/llama-server"
            log_info "Run: ./scripts/rebuild.sh"
            exit 1
        fi
        log_ok "$be binary: $bin_dir/llama-server"
    fi
done

# Filter models if specific one requested
[[ -n "$TEST_MODEL" ]] && MODELS=("$TEST_MODEL")

# Ensure clean state
trap 'stop_server' EXIT

# Run benchmarks
for be in rocm vulkan; do
    if [[ "$BACKEND" == "both" ]] || [[ "$BACKEND" == "$be" ]]; then
        mkdir -p "$BENCH_DIR/$be"

        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  $be backend"
        echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        summaries=()

        for model_entry in "${MODELS[@]}"; do
            IFS=':' read -r model extra_flags <<< "$model_entry"

            if [[ ! -f "$MODEL_DIR/$model" ]]; then
                log_warn "Skipping $model (not found)"
                continue
            fi

            summary=$(run_model_benchmark "$model" "$extra_flags" "$be") || {
                log_error "Benchmark failed for $model on $be"
                continue
            }
            summaries+=("$summary")
        done

        if [[ ${#summaries[@]} -gt 0 ]]; then
            generate_aggregate "$be" "${summaries[@]}"
        fi
    fi
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Benchmark Complete"
echo "  Results: $BENCH_DIR"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
