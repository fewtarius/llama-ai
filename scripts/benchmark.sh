#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Llama.cpp benchmark suite.
#
# Runs three test categories against every (or one specific) model in
# $MODEL_DIR on one or both backends:
#
#   cache    Cold/warm prompt-cache test at three prompt sizes. Starts a
#            llama-server, sends the same prompt twice (cache cleared then
#            restored), captures prompt-eval TTFT and decode speedup, plus
#            MoE expert residency hit-rate when the profile uses residency.
#
#   bench    llama-bench sweep over prompt-processing sizes (512/2k/8k/16k)
#            and text-generation sizes (128/256/512/1k/2k). The canonical
#            "tokens/sec at this batch" numbers users see quoted everywhere.
#
#   batched  llama-batched-bench sweep over parallel slot counts (1/2/4/8).
#            Answers "how does this model perform with concurrent users?"
#
# Per-test drilldown files (raw.json, report.md, server logs) live under
# <backend>/<model>/<test>/. A combined per-model summary.md sits at
# <backend>/<model>/summary.md, and an index.md leaderboard at
# <backend>/index.md.
#
# By default, prior runs are overwritten. Pass --archive to move existing
# results to <backend>/archive/<timestamp>/ before re-running.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCH_ROOT="$PROJECT_ROOT/benchmarks"
MODEL_DIR="$PROJECT_ROOT/models"
SSD_CACHE_DIR="$PROJECT_ROOT/ssd-cache"
SCRATCH_DIR="$PROJECT_ROOT/scratch"

ROCM_BIN="$PROJECT_ROOT/src/cachy-llama-rocm/build/bin"
VULKAN_BIN="$PROJECT_ROOT/src/cachy-llama-vulkan/build/bin"

BACKEND="vulkan"
PORT=9090
NGL=99
MAX_TOKENS=128
TEST_MODEL=""
ARCHIVE=0
ARCHIVE_DOC_ONLY=1   # sentinel; --archive is removed but flag kept for compat
TESTS="cache,bench,batched"
BENCH_TIMEOUT=900

PROMPT_SIZES=(
    "small:4096"
    "medium:20480"
    "large:61440"
)
PROMPT_INSTRUCTION="Summarize this passage in one sentence."

BENCH_PP_SIZES=(512 2048 8192 16384)
BENCH_TG_SIZES=(128 256 512 1024 2048)
BENCH_REPETITIONS=5

BATCHED_PARALLEL=(1 2 4 8)
BATCHED_PROMPT_SIZE=2048
BATCHED_GEN_SIZE=128
BATCHED_REPETITIONS=3

GUTENBERG_URL="http://aleph.gutenberg.org/cache/epub/1184/pg1184.txt"
GUTENBERG_CACHE="$SCRATCH_DIR/pg1184.txt"

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

# shellcheck source=scripts/lib-discover-models.sh
source "$SCRIPT_DIR/lib-discover-models.sh"

# =============================================================================
# Arg parsing
# =============================================================================
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --backend BACKEND       rocm, vulkan, or both (default: vulkan)
    --model MODEL           Test a specific model only (bypasses discovery)
    --tests TESTS           Comma-separated: cache,bench,batched (default: all)
    --port PORT             Server port for cache test (default: 9090)
    --ctx SIZE              Override context size (default: per-profile)
    --ngl LAYERS            Override GPU layers (default: 99)
    --tokens N              Max output tokens for cache test (default: 128)
    --repetitions N         llama-bench repetitions (default: 5)
    --parallel COUNTS       Comma-separated batched parallel counts (default: 1,2,4,8)
    --help                  Show this help

OUTPUT:
    benchmarks/<backend>/index.md             # leaderboard
    benchmarks/<backend>/<model>/
    ├── summary.md                            # combined: cache + bench + batched
    ├── summary.json
    ├── cache/                                # cold/warm prompt tests
    │   ├── server-*-cold.log, server-*-warm.log
    │   ├── *-result.json
    │   ├── summary.json
    │   └── analysis.md                       # log_analyzer.py output
    ├── bench/                                # llama-bench
    │   ├── raw.json                          # JSON output from llama-bench -o json
    │   ├── summary.json
    │   └── report.md
    └── batched/                              # llama-batched-bench
        ├── raw.json
        ├── summary.json
        └── report.md

EXAMPLES:
    $(basename "$0")                                  # all models, all tests, vulkan
    $(basename "$0") --tests bench                    # only llama-bench sweep
    $(basename "$0") --model Qwen3-...A3B-UD-Q4_K_XL.gguf
    $(basename "$0") --backend both                   # vulkan + rocm
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)    BACKEND="$2"; shift 2 ;;
        --model)      TEST_MODEL="$2"; shift 2 ;;
        --tests)      TESTS="$2"; shift 2 ;;
        --archive)
            log_warn "--archive is deprecated (git history preserves prior runs); ignoring"
            shift ;;
        --port)       PORT="$2"; shift 2 ;;
        --ctx)        CTX_OVERRIDE="$2"; shift 2 ;;
        --ngl)        NGL="$2"; shift 2 ;;
        --tokens)     MAX_TOKENS="$2"; shift 2 ;;
        --repetitions)BENCH_REPETITIONS="$2"; shift 2 ;;
        --parallel)   BATCHED_PARALLEL_RAW="$2"; shift 2 ;;
        --help|-h)    usage; exit 0 ;;
        *)            log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

IFS=',' read -ra SELECTED_TESTS <<< "$TESTS"
for t in "${SELECTED_TESTS[@]}"; do
    case "$t" in
        cache|bench|batched) ;;
        *) log_error "Unknown test: $t (expected: cache,bench,batched)"; exit 1 ;;
    esac
done
log_info "Tests: ${SELECTED_TESTS[*]}"

[[ "$BACKEND" != "rocm" && "$BACKEND" != "vulkan" && "$BACKEND" != "both" ]] \
    && { log_error "Invalid backend: $BACKEND"; usage; exit 1; }

if [[ -n "${BATCHED_PARALLEL_RAW:-}" ]]; then
    IFS=',' read -ra BATCHED_PARALLEL <<< "$BATCHED_PARALLEL_RAW"
fi

# =============================================================================
# Generic helpers (must be defined before any code that calls them)
# =============================================================================

get_binary() {
    if [[ "$1" == "vulkan" ]]; then echo "$VULKAN_BIN"; else echo "$ROCM_BIN"; fi
}

get_model_size_bytes() {
    local path="$MODEL_DIR/$1"
    [[ -f "$path" ]] || { echo 0; return; }
    stat -c %s "$path" 2>/dev/null || stat -f %z "$path" 2>/dev/null || echo 0
}

scale_timeout_for_model() {
    local model="$1" base_timeout="$2"
    local size_bytes size_gb model_mult=1
    size_bytes=$(get_model_size_bytes "$model")
    size_gb=$(( size_bytes / 1024 / 1024 / 1024 ))
    if   [[ $size_gb -ge 50 ]]; then model_mult=4
    elif [[ $size_gb -ge 30 ]]; then model_mult=3
    elif [[ $size_gb -ge 20 ]]; then model_mult=2
    fi
    echo $(( base_timeout * model_mult ))
}

setup_backend_env() {
    local backend="$1"
    source "$PROJECT_ROOT/scripts/env.sh" "$backend"
    source "$PROJECT_ROOT/scripts/detect-gpu.sh"
    [[ "$backend" == "rocm" ]] && export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
}

get_model_profile() {
    local model="$1" backend="$2"
    ./llama-run.sh --print-profile --server --model "$MODEL_DIR/$model" --backend "$backend" 2>/dev/null \
        | grep -E '^[A-Z_][A-Z0-9_]*=' || return 1
}

# =============================================================================
# Discovery + binary check
# =============================================================================
mapfile -t MODELS < <(discover_models "$MODEL_DIR")

log_info "Discovered ${#MODELS[@]} benchmarkable model(s) in $MODEL_DIR"

if [[ ${#MODELS[@]} -eq 0 ]]; then
    log_error "No .gguf models found in $MODEL_DIR"
    exit 1
fi

[[ -n "$TEST_MODEL" ]] && MODELS=("$TEST_MODEL")

for be in rocm vulkan; do
    [[ "$BACKEND" == "both" || "$BACKEND" == "$be" ]] || continue
    bin_dir=$(get_binary "$be")
    if [[ ! -x "$bin_dir/llama-server" ]]; then
        log_error "$be binary not found: $bin_dir/llama-server"
        log_info "Run: ./scripts/rebuild.sh"
        exit 1
    fi
    if [[ "$TESTS" == *"bench"* && ! -x "$bin_dir/llama-bench" ]]; then
        log_error "$be: llama-bench not found: $bin_dir/llama-bench"
        exit 1
    fi
    if [[ "$TESTS" == *"batched"* && ! -x "$bin_dir/llama-batched-bench" ]]; then
        log_error "$be: llama-batched-bench not found: $bin_dir/llama-batched-bench"
        exit 1
    fi
    log_ok "$be binaries present"
done

# =============================================================================
# Output path helpers
# =============================================================================

model_dir() {
    echo "$BENCH_ROOT/$1/$2"
}

test_dir() {
    echo "$(model_dir "$1" "$2")/$3"
}

# State preserved by --archive flag in earlier revisions. Removed: this is
# a git repo and benchmarks/ is gitignored, so prior results are kept via
# git history (commit benchmarks/ between runs to compare).

# =============================================================================
# Test: cache
# =============================================================================

build_prompt() {
    local size_bytes="$1" text_file="$2"
    local prefix
    prefix=$(head -c "$size_bytes" <(sed '1s/^\xEF\xBB\xBF//' "$text_file"))
    python3 -c "
import json, sys
text = sys.stdin.read()
prompt = text + '\n\n' + '$PROMPT_INSTRUCTION'
print(json.dumps(prompt))
" <<< "$prefix"
}

SERVER_PID=""
SERVER_LOG=""

wait_for_server() {
    local max_attempts="${1:-900}"
    local attempt=0
    printf "    Waiting for server"
    while [[ $attempt -lt $max_attempts ]]; do
        local resp http_code body
        resp=$(curl -s -w "\n%{http_code}" "http://localhost:$PORT/v1/models" 2>/dev/null) || true
        http_code=$(echo "$resp" | tail -1)
        body=$(echo "$resp" | head -n -1)
        if [[ "$http_code" == "200" ]] && echo "$body" | grep -q "gguf"; then
            printf "\n"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf "\n"
            log_error "Server exited unexpectedly (PID $SERVER_PID)"
            tail -20 "$SERVER_LOG" 2>/dev/null || true
            return 1
        fi
        if [[ -f "$SERVER_LOG" ]]; then
            local fatal
            fatal=$(grep -iE "SIGSEGV|SIGABRT|segfault|segmentation fault|killed \(signal|out of memory|Out of memory|Killed process" "$SERVER_LOG" 2>/dev/null | tail -1 || echo "")
            if [[ -n "$fatal" ]]; then
                printf "\n"
                log_error "Server crashed: $fatal"
                return 1
            fi
        fi
        attempt=$((attempt + 1))
        [[ $((attempt % 5)) -eq 0 ]] && printf "."
        [[ $((attempt % 30)) -eq 0 ]] && printf " %ds\n" "$attempt"
        sleep 1
    done
    printf "\n"
    log_error "Server did not respond after ${max_attempts}s"
    return 1
}

start_server() {
    local model="$1" extra_flags="$2" backend="$3" ssd_path="$4" log_path="$5"

    log_info "Starting server: $model (backend: $backend, ssd: ${ssd_path:-none})"

    pkill -9 llama-server 2>/dev/null || true
    sleep 3

    local profile
    profile=$(get_model_profile "$model" "$backend") || {
        log_error "Failed to get profile from llama-run.sh"
        return 1
    }
    eval "$profile"

    setup_backend_env "$backend"

    local llama_bin
    llama_bin=$(get_binary "$backend")

    local cmd=("$llama_bin/llama-server")
    cmd+=(-m "$MODEL_PATH")
    [[ -n "${CTX_OVERRIDE:-}" ]] && cmd+=(-c "$CTX_OVERRIDE") || cmd+=(-c "$CTX_SIZE")
    cmd+=(-ngl "$GPU_LAYERS")
    cmd+=(--threads "$THREADS" --threads-batch "$THREADS")
    cmd+=(--port "$PORT" --host 0.0.0.0)
    [[ -n "$OVERRIDE_BATCH_SIZE" ]] && cmd+=($OVERRIDE_BATCH_SIZE)
    cmd+=(--cache-type-k "$KV_CACHE_TYPE_K" --cache-type-v "$KV_CACHE_TYPE_V")
    cmd+=(-fa on --jinja)
    [[ -n "${OVERRIDE_REASONING:-}" ]] && cmd+=(--reasoning "$OVERRIDE_REASONING")
    cmd+=(--slot-prompt-similarity 0.20)
    cmd+=(--slot-save-path "$SSD_CACHE_DIR")
    cmd+=(--kv-unified)
    cmd+=(-np 1 --prio 3 --prio-batch 3 --metrics)
    cmd+=(--ctx-checkpoints "${SSD_CHECKPOINTS:-8}" --cache-reuse 512)

    if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
        IFS=' ' read -ra PROFILE_ARGS <<< "$EXTRA_SERVER_ARGS"
        cmd+=("${PROFILE_ARGS[@]}")
    fi

    if [[ -n "$ssd_path" ]]; then
        mkdir -p "$ssd_path"
        cmd+=(--cache-ssd "$ssd_path")
        [[ -n "${SSD_CHECKPOINTS:-}" ]] && cmd+=(--cache-ssd-checkpoints "$SSD_CHECKPOINTS")
        [[ -n "${SSD_HOT_WINDOW:-}" ]]  && cmd+=(--cache-ssd-hot-window "$SSD_HOT_WINDOW")
        [[ -n "${SSD_WARM_WINDOW:-}" ]] && cmd+=(--cache-ssd-warm-window "$SSD_WARM_WINDOW")
        [[ -n "${SSD_MAX_COLD:-}" ]]    && cmd+=(--cache-ssd-max-cold "$SSD_MAX_COLD")
        [[ -n "${SSD_PAGE_SIZE:-}" ]]   && cmd+=(--cache-ssd-page-size "$SSD_PAGE_SIZE")
        [[ -n "${SSD_HOT_RAM:-}" ]]     && cmd+=(--cache-ssd-hot-ram "$SSD_HOT_RAM")
        [[ -n "${SSD_WARM_RAM:-}" ]]    && cmd+=(--cache-ssd-warm-ram "$SSD_WARM_RAM")
    fi

    [[ -n "$extra_flags" ]] && IFS=' ' read -ra FLAGS <<< "$extra_flags" && cmd+=("${FLAGS[@]}")

    {
        printf "BENCHMARK COMMAND:"
        for arg in "${cmd[@]}"; do printf " %q" "$arg"; done
        printf "\n"
    } > "${log_path}.cmd"

    "${cmd[@]}" > "$log_path" 2>&1 &
    SERVER_PID=$!

    local model_bytes wait_timeout=900
    model_bytes=$(get_model_size_bytes "$model")
    if   [[ $model_bytes -ge $((30 * 1024 * 1024 * 1024)) ]]; then wait_timeout=2700
    elif [[ $model_bytes -ge $((20 * 1024 * 1024 * 1024)) ]]; then wait_timeout=1800
    fi

    if ! wait_for_server "$wait_timeout"; then
        log_error "Server failed to start"
        tail -30 "$log_path"
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

call_api() {
    local prompt_json="$1" run_label="$2" out_dir="$3" timeout="${4:-$BENCH_TIMEOUT}"

    local raw_resp_file="$out_dir/${run_label}-response.json"
    local stats_file="$out_dir/${run_label}-stats.json"
    local start_ns end_ns wall_ms http_code

    start_ns=$(date +%s%N)
    http_code=$(curl -s --max-time "$timeout" \
        -w "%{http_code}" \
        -o "$raw_resp_file" \
        "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$prompt_json}],\"max_tokens\":$MAX_TOKENS}") || {
        echo "{\"error\":\"curl failed (timeout ${timeout}s)\",\"wall_ms\":0}" > "$stats_file"
        return 1
    }
    end_ns=$(date +%s%N)
    wall_ms=$(( (end_ns - start_ns) / 1000000 ))

    if [[ "$http_code" != "200" ]]; then
        local body_preview
        body_preview=$(head -c 500 "$raw_resp_file" 2>/dev/null || echo "(empty)")
        python3 -c "
import json,sys
print(json.dumps({'error':'HTTP $http_code','wall_ms':$wall_ms,'body_preview':sys.stdin.read()}))" <<< "$body_preview" > "$stats_file"
        return 1
    fi

    python3 - "$raw_resp_file" "$wall_ms" "$stats_file" << 'PYEOF'
import json, sys

resp_file, wall_ms, stats_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]

try:
    with open(resp_file) as f:
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
}

detect_cache_state() {
    local log_file="$1"
    python3 - "$log_file" << 'PYEOF'
import re, json, sys
text = open(sys.argv[1]).read()
sys_cache_hit = bool(re.search(r'system prompt cache hit', text))
sys_prompt_restored = bool(re.search(r'restored system prompt from cache', text))
loaded_match = re.search(r'loaded ([0-9]+) checkpoints from', text)
loaded_checkpoints = loaded_match is not None and int(loaded_match.group(1)) > 0
cold_restored = bool(re.search(r'cold-start restored SSD', text))
warm_restored = bool(re.search(r'restored in-memory checkpoint', text))
stored = len(re.findall(r'SSD cache: stored checkpoint', text))
divergences = len(re.findall(r'cache prefix divergence', text))
if cold_restored:
    state = 'ssd_cold'
elif warm_restored or sys_cache_hit or sys_prompt_restored or loaded_checkpoints:
    state = 'ssd_warm'
else:
    state = 'miss'
total_checkpoints = len(re.findall(r'SSD cache: (stored|promoted|demoted) checkpoint', text))
print(json.dumps({
    'cache_state': state,
    'ssd_stored': stored,
    'divergences': divergences,
    'total_checkpoints': total_checkpoints,
}))
PYEOF
}

detect_moe_residency_stats() {
    local log_file="$1"
    python3 - "$log_file" << 'PYEOF'
import re, json, sys
text = open(sys.argv[1]).read()
enabled = bool(re.search(r'moe-residency: enabled for', text))
stats = re.findall(
    r'moe-residency:\s+decodes=(\d+)\s+touches=(\d+)\s+hits=(\d+)\s+misses=(\d+)\s+evictions=(\d+)\s+hit_rate=([\d.]+)%',
    text,
)
result = {'enabled': enabled, 'present': bool(stats)}
if stats:
    decodes, touches, hits, misses, evictions, hit_rate = stats[-1]
    result.update({
        'decodes': int(decodes), 'touches': int(touches), 'hits': int(hits),
        'misses': int(misses), 'evictions': int(evictions), 'hit_rate_pct': float(hit_rate),
    })
act_match = re.search(
    r'moe-residency:\s+enabled for\s+(\d+)\s+MoE layers\s+\((\d+)\s+experts,\s+(\d+)\s+used/token\)',
    text,
)
if act_match:
    result['moe_layers'] = int(act_match.group(1))
    result['n_expert'] = int(act_match.group(2))
    result['n_expert_used'] = int(act_match.group(3))
print(json.dumps(result))
PYEOF
}

detect_profile() {
    local cmd_file="$1"
    python3 - "$cmd_file" << 'PYEOF'
import re, json, os, sys
if not os.path.exists(sys.argv[1]):
    print(json.dumps({'profile': 'unknown'})); exit()
text = open(sys.argv[1]).read()
profile = 'unknown'; features = []
if '--moe-expert-residency' in text:
    profile = 'moe-optimized'; features.append('moe-residency')
elif re.search(r'--ctx-checkpoints 64', text):
    profile = 'dense-large'
elif re.search(r'--cache-ram 16384', text):
    profile = 'halo-tier'
elif re.search(r'--cache-ram 4096', text):
    profile = 'standard'
else:
    profile = 'default'
if '--no-checkpoint-near-end' in text: features.append('no-checkpoint-near-end')
if '--kv-unified' in text: features.append('kv-unified')
if '--cpu-moe' in text or '--cmoe' in text: features.append('cpu-moe')
m = re.search(r'--cache-ssd-cold-maxsize\s+(\d+)', text)
if m: features.append(f'ssd-cold-maxsize={m.group(1)}MiB')
print(json.dumps({'profile': profile, 'features': features}))
PYEOF
}

measure_ssd_cache() {
    local path="$1"
    python3 - "$path" << 'PYEOF'
import os, json, sys
total = 0; file_count = 0
try:
    for root, dirs, files in os.walk(sys.argv[1]):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                file_count += 1
            except OSError:
                pass
except OSError:
    pass
print(json.dumps({'bytes': total, 'mib': round(total / 1024 / 1024, 1), 'files': file_count}))
PYEOF
}

run_cache_test() {
    local model="$1" extra_flags="$2" backend="$3" out_dir="$4"
    local model_name
    model_name=$(basename "$model" .gguf)

    log_header "cache test: $model_name"

    for size_entry in "${PROMPT_SIZES[@]}"; do
        IFS=':' read -r size_label size_bytes <<< "$size_entry"
        printf "  ${YELLOW}▶ %s (%.0f KB)${NC}\n" "$size_label" "$(echo "$size_bytes/1024" | bc -l)"

        local prompt_json timeout cold_pass=true warm_pass=true
        prompt_json=$(build_prompt "$size_bytes" "$GUTENBERG_CACHE")
        timeout=$BENCH_TIMEOUT
        if   [[ $size_bytes -ge 40000 ]]; then timeout=$((BENCH_TIMEOUT * 3))
        elif [[ $size_bytes -ge 15000 ]]; then timeout=$((BENCH_TIMEOUT * 2))
        fi
        timeout=$(scale_timeout_for_model "$model" "$timeout")

        rm -rf "$SSD_CACHE_DIR"
        local cold_log="$out_dir/server-${size_label}-cold.log"
        if ! start_server "$model" "$extra_flags" "$backend" "$SSD_CACHE_DIR" "$cold_log"; then
            cold_pass=false; stop_server; continue
        fi
        printf "    cold: " >&2
        call_api "$prompt_json" "${size_label}-cold" "$out_dir" "$timeout" > /dev/null || cold_pass=false
        local cold_stats="$out_dir/${size_label}-cold-stats.json"
        if $cold_pass && [[ -f "$cold_stats" ]]; then
            local pt ttft
            pt=$(python3 -c "import json; print(json.load(open('$cold_stats')).get('prompt_tokens', 0))" 2>/dev/null || echo 0)
            ttft=$(python3 -c "import json; print(json.load(open('$cold_stats')).get('ttft_ms', 0))" 2>/dev/null || echo 0)
            printf "${GREEN}%s tokens, TTFT %sms${NC}\n" "$pt" "$ttft" >&2
        else
            printf "${RED}failed${NC}\n" >&2
        fi
        local cache_cold moe_cold profile_cold
        cache_cold=$(detect_cache_state "$cold_log") || cache_cold='{"cache_state":"unknown"}'
        moe_cold=$(detect_moe_residency_stats "$cold_log") || moe_cold='{"enabled":false}'
        profile_cold=$(detect_profile "$cold_log.cmd") || profile_cold='{"profile":"unknown"}'
        stop_server

        local warm_log="$out_dir/server-${size_label}-warm.log"
        if ! start_server "$model" "$extra_flags" "$backend" "$SSD_CACHE_DIR" "$warm_log"; then
            warm_pass=false; stop_server; continue
        fi
        printf "    warm: " >&2
        call_api "$prompt_json" "${size_label}-warm" "$out_dir" "$timeout" > /dev/null || warm_pass=false
        local warm_stats="$out_dir/${size_label}-warm-stats.json"
        if $warm_pass && [[ -f "$warm_stats" ]]; then
            local pt ttft
            pt=$(python3 -c "import json; print(json.load(open('$warm_stats')).get('prompt_tokens', 0))" 2>/dev/null || echo 0)
            ttft=$(python3 -c "import json; print(json.load(open('$warm_stats')).get('ttft_ms', 0))" 2>/dev/null || echo 0)
            printf "${GREEN}%s tokens, TTFT %sms${NC}\n" "$pt" "$ttft" >&2
        else
            printf "${RED}failed${NC}\n" >&2
        fi
        local cache_warm moe_warm ssd_size
        cache_warm=$(detect_cache_state "$warm_log") || cache_warm='{"cache_state":"unknown"}'
        moe_warm=$(detect_moe_residency_stats "$warm_log") || moe_warm='{"enabled":false}'
        stop_server

        ssd_size=$(measure_ssd_cache "$SSD_CACHE_DIR") || ssd_size='{"mib":0,"files":0}'

        local result_file="$out_dir/${size_label}-result.json"
        python3 - "$out_dir" "$size_label" "$size_bytes" "$cold_pass" "$warm_pass" \
            "$cache_cold" "$cache_warm" "$moe_cold" "$moe_warm" "$profile_cold" "$ssd_size" "$result_file" << 'PYEOF'
import json, sys, os
out_dir, size_label, size_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3])
cold_pass = sys.argv[4] == 'true'
warm_pass = sys.argv[5] == 'true'
cache_cold = json.loads(sys.argv[6]); cache_warm = json.loads(sys.argv[7])
moe_cold = json.loads(sys.argv[8]);   moe_warm = json.loads(sys.argv[9])
profile = json.loads(sys.argv[10]);   ssd_size = json.loads(sys.argv[11])
result_file = sys.argv[12]

def load(label):
    try: return json.load(open(os.path.join(out_dir, f'{label}-stats.json')))
    except Exception as e: return {'error': str(e)}
cold = load(f'{size_label}-cold') if cold_pass else {'error':'failed'}
warm = load(f'{size_label}-warm') if warm_pass else {'error':'failed'}

cpm = cold.get('prompt_ms', 0); wpm = warm.get('prompt_ms', 0)
cwall = cold.get('wall_ttft_ms', 0); wwall = warm.get('wall_ttft_ms', 0)
result = {
    'size_label': size_label, 'size_bytes': size_bytes,
    'cold': cold, 'warm': warm,
    'cache_cold': cache_cold, 'cache_warm': cache_warm,
    'moe_cold': moe_cold, 'moe_warm': moe_warm,
    'profile': profile, 'ssd_size': ssd_size,
    'prompt_eval_speedup': round(cpm / wpm, 2) if wpm > 0 and cpm > 0 else 0,
    'ttft_speedup': round(cwall / wwall, 2) if wwall > 0 and cwall > 0 else 0,
    'cold_prompt_tps': cold.get('prompt_tps', 0),
    'warm_prompt_tps': warm.get('prompt_tps', 0),
    'cold_ppt_ms': cold.get('prompt_per_token_ms', 0),
    'warm_ppt_ms': warm.get('prompt_per_token_ms', 0),
    'cold_gen_ppt_ms': cold.get('predicted_per_token_ms', 0),
    'warm_gen_ppt_ms': warm.get('predicted_per_token_ms', 0),
    'cold_ttft_ms': cold.get('ttft_ms', 0),
    'warm_ttft_ms': warm.get('ttft_ms', 0),
    'cold_wall_ttft_ms': cwall,
    'warm_wall_ttft_ms': wwall,
}
with open(result_file, 'w') as f: json.dump(result, f, indent=2)
PYEOF
    done

    # Render per-test drilldown using log_analyzer.py on the warm logs.
    python3 - "$out_dir" "$model_name" "$backend" << 'PYEOF'
import json, os, sys, glob
sys.path.insert(0, os.path.join(os.environ.get('PROJECT_ROOT', '/home/deck/llama-ai'), 'scripts'))
import log_analyzer

out_dir = sys.argv[1]
model_name = sys.argv[2]
backend = sys.argv[3]

results = []
for rf in sorted(glob.glob(os.path.join(out_dir, '*-result.json'))):
    try: results.append(json.load(open(rf)))
    except Exception: pass

summary = {'model': model_name, 'backend': backend, 'results': results}
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

md_parts = [f'# {model_name} — Cache Test ({backend})', '']
for rf in sorted(glob.glob(os.path.join(out_dir, '*-result.json'))):
    label = os.path.basename(rf).replace('-result.json', '')
    warm_log = os.path.join(out_dir, f'server-{label}-warm.log')
    if os.path.exists(warm_log):
        md_parts.append(f'## Warm Run: {label} prompt')
        md_parts.append('')
        metrics = log_analyzer.analyze(warm_log)
        md_parts.append(log_analyzer.render_markdown(metrics))
        md_parts.append('')

md_parts.append('## Cold vs Warm Speedup')
md_parts.append('')
md_parts.append('| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Cache | MoE hit |')
md_parts.append('|------|-----------|-----------|--------------|-------------|-------------|-------|---------|')
for r in results:
    if r.get('error'):
        md_parts.append(f'| {r.get("size_label","?")} | - | - | - | - | - | error | - |')
        continue
    cw = r.get('cache_warm', {})
    moe = r.get('moe_warm', {})
    moe_hit = f"{moe.get('hit_rate_pct', 0):.1f}%" if moe.get('present') else '-'
    md_parts.append(
        f'| {r["size_label"]} | {r.get("cold_ttft_ms", 0)}ms | {r.get("warm_ttft_ms", 0)}ms | '
        f'{r.get("prompt_eval_speedup", 0)}x | {r.get("cold_ppt_ms", 0)}ms | {r.get("warm_ppt_ms", 0)}ms | '
        f'{cw.get("cache_state", "?")} | {moe_hit} |'
    )

md_parts.append('')
md_parts.append('**Cache states:** `ssd_cold` = restored from SSD after restart, '
                '`ssd_warm` = in-memory checkpoint or system cache hit, '
                '`miss` = no cache hit')
md_parts.append('**TTFT** = Time To First Token (server-side prompt eval)')
md_parts.append('**TTFT Speedup** = cold TTFT / warm TTFT')
md_parts.append('**MoE hit** = warm-run MoE expert residency hit rate (only for MoE with --moe-expert-residency)')

with open(os.path.join(out_dir, 'analysis.md'), 'w') as f:
    f.write('\n'.join(md_parts))
PYEOF
}

# =============================================================================
# Test: bench (llama-bench)
# =============================================================================

run_bench_test() {
    local model="$1" backend="$2" out_dir="$3"
    local model_name
    model_name=$(basename "$model" .gguf)

    log_header "bench test: $model_name"

    local profile
    profile=$(get_model_profile "$model" "$backend") || {
        log_error "Failed to get profile from llama-run.sh"; return 1
    }
    eval "$profile"
    setup_backend_env "$backend"

    local llama_bin bench_bin
    llama_bin=$(get_binary "$backend")
    bench_bin="$llama_bin/llama-bench"
    [[ -x "$bench_bin" ]] || { log_error "llama-bench not found: $bench_bin"; return 1; }

    # Build pp+tg pairs. -pg PP,TG runs one test. pp=0 means pure
    # generation, tg=0 means pure prefill -- the data we actually want.
    # -p/-n defaults are overridden to 0 so we only run the requested
    # matrix and not a redundant "512 prompt, 128 gen" baseline.
    local batch_size=2048 ubatch_size=512
    if [[ -n "${OVERRIDE_BATCH_SIZE:-}" ]]; then
        # OVERRIDE_BATCH_SIZE is "--batch-size X --ubatch-size Y".
        local bsz ubsz
        read -r _ bsz _ _ ubsz <<< "$OVERRIDE_BATCH_SIZE"
        [[ -n "$bsz"  ]] && batch_size="$bsz"
        [[ -n "$ubsz" ]] && ubatch_size="$ubsz"
    fi

    local ctx_value="${CTX_OVERRIDE:-$CTX_SIZE}"

    local cmd=("$bench_bin"
        -m "$MODEL_PATH"
        -p 0 -n 0
        -r "$BENCH_REPETITIONS"
        -ngl "$GPU_LAYERS"
        -t "$THREADS"
        -ctk "$KV_CACHE_TYPE_K" -ctv "$KV_CACHE_TYPE_V"
        -fa on
        -b "$batch_size" -ub "$ubatch_size"
        -o json
        --progress
    )
    # Pure-pp sweep: tg=0
    for pp in "${BENCH_PP_SIZES[@]}"; do cmd+=( -pg "$pp,0" ); done
    # Pure-tg sweep: pp=0
    for tg in "${BENCH_TG_SIZES[@]}"; do cmd+=( -pg "0,$tg" ); done

    log_info "llama-bench sweep: ${#BENCH_PP_SIZES[@]} pp sizes + ${#BENCH_TG_SIZES[@]} tg sizes x $BENCH_REPETITIONS reps"

    local raw_file="$out_dir/raw.json"
    local stderr_file="$out_dir/stderr.log"

    if ! "${cmd[@]}" > "$raw_file" 2> "$stderr_file"; then
        log_error "llama-bench failed for $model_name"
        tail -20 "$stderr_file" >&2
        return 1
    fi

    if ! python3 -c "import json; json.load(open('$raw_file'))" 2>/dev/null; then
        log_error "llama-bench JSON output invalid for $model_name"
        return 1
    fi

    python3 - "$raw_file" "$BENCH_REPETITIONS" "$model_name" "$backend" "$out_dir/report.md" << 'PYEOF'
import json, sys
raw_file, reps, model_name, backend, report_file = sys.argv[1:6]

data = json.load(open(raw_file))

def fmt(x, suf=''):
    try: return f'{float(x):.1f}{suf}'
    except (TypeError, ValueError): return '-'

# Pure pp (n_gen=0): avg_ts IS pp t/s. Pure tg (n_prompt=0): avg_ts IS tg t/s.
pp_rows = sorted([r for r in data if r.get('n_gen') == 0],
                 key=lambda r: r['n_prompt'])
tg_rows = sorted([r for r in data if r.get('n_prompt') == 0],
                 key=lambda r: r['n_gen'])

first = data[0] if data else {}
md = [f'# {model_name} -- llama-bench ({backend})', '']
md.append(f'**Repetitions per test:** {reps}  ')
md.append(f'**Model params:** {fmt(first.get("model_n_params", 0) / 1e9)}B  ')
md.append(f'**Model size on disk:** {fmt(first.get("model_size", 0) / 1024**3)} GiB  ')
md.append('')

md.append('## Prompt Processing (tg=0, pure prefill)')
md.append('')
md.append('| pp tokens | avg ms | pp t/s | sd (t/s) |')
md.append('|----------:|-------:|-------:|---------:|')
for r in pp_rows:
    avg_ms = r.get('avg_ns', 0) / 1e6
    md.append(f'| {r["n_prompt"]} | {fmt(avg_ms)} | {fmt(r.get("avg_ts"))} | {fmt(r.get("stddev_ts"))} |')

md.append('')
md.append('## Text Generation (pp=0, pure generation)')
md.append('')
md.append('| tg tokens | avg ms | tg t/s | sd (t/s) |')
md.append('|----------:|-------:|-------:|---------:|')
for r in tg_rows:
    avg_ms = r.get('avg_ns', 0) / 1e6
    md.append(f'| {r["n_gen"]} | {fmt(avg_ms)} | {fmt(r.get("avg_ts"))} | {fmt(r.get("stddev_ts"))} |')

md.append('')
md.append(f'_Source: [`raw.json`](./raw.json)_')
md.append('')

with open(report_file, 'w') as f: f.write('\n'.join(md))
PYEOF

    python3 - "$raw_file" "$out_dir/summary.json" << 'PYEOF'
import json, sys
raw_file, summary_file = sys.argv[1], sys.argv[2]
data = json.load(open(raw_file))

pp_rows = [r for r in data if r.get('n_gen') == 0]
tg_rows = [r for r in data if r.get('n_prompt') == 0]

def find(rows, key, val, want):
    for r in rows:
        if r.get(key) == val: return r.get(want)
    return None

summary = {
    'pp_tps': {str(r['n_prompt']): r.get('avg_ts', 0) for r in pp_rows},
    'tg_tps': {str(r['n_gen']): r.get('avg_ts', 0) for r in tg_rows},
    'pp_2048_tps': find(pp_rows, 'n_prompt', 2048, 'avg_ts') or 0,
    'pp_8192_tps': find(pp_rows, 'n_prompt', 8192, 'avg_ts') or 0,
    'tg_128_tps': find(tg_rows, 'n_gen', 128, 'avg_ts') or 0,
    'tg_512_tps': find(tg_rows, 'n_gen', 512, 'avg_ts') or 0,
    'model_size_gib': round(data[0].get('model_size', 0) / 1024**3, 2) if data else 0,
    'model_params_b': round(data[0].get('model_n_params', 0) / 1e9, 2) if data else 0,
}
with open(summary_file, 'w') as f: json.dump(summary, f, indent=2)
PYEOF

    log_ok "bench complete: $raw_file"
}

# =============================================================================
# Test: batched (llama-batched-bench)
# =============================================================================

run_batched_test() {
    local model="$1" backend="$2" out_dir="$3"
    local model_name
    model_name=$(basename "$model" .gguf)

    log_header "batched test: $model_name"

    local profile
    profile=$(get_model_profile "$model" "$backend") || {
        log_error "Failed to get profile from llama-run.sh"; return 1
    }
    eval "$profile"
    setup_backend_env "$backend"

    local llama_bin batched_bin
    llama_bin=$(get_binary "$backend")
    batched_bin="$llama_bin/llama-batched-bench"
    [[ -x "$batched_bin" ]] || { log_error "llama-batched-bench not found: $batched_bin"; return 1; }

    local ctx_value="${CTX_OVERRIDE:-$CTX_SIZE}"

    local all_results=()
    local ctx_value="${CTX_OVERRIDE:-$CTX_SIZE}"

    for np in "${BATCHED_PARALLEL[@]}"; do
        log_info "  parallel=$np"
        # ctx must be >= pp + np * tg so all slots fit; otherwise the
        # decode slot allocation fails and the test prints no row.
        local needed_ctx=$((BATCHED_PROMPT_SIZE + np * BATCHED_GEN_SIZE))
        local run_ctx=$ctx_value
        [[ $run_ctx -lt $needed_ctx ]] && run_ctx=$needed_ctx

        local cmd=("$batched_bin"
            -m "$MODEL_PATH"
            -c "$run_ctx"
            -b 2048 -ub 512
            -npp "$BATCHED_PROMPT_SIZE"
            -ntg "$BATCHED_GEN_SIZE"
            -npl "$np"
            -ngl "$GPU_LAYERS"
            -t "$THREADS"
            -ctk "$KV_CACHE_TYPE_K" -ctv "$KV_CACHE_TYPE_V"
            -fa on
            --output-format jsonl
        )

        local per_file="$out_dir/np${np}.json"
        local per_stderr="$out_dir/np${np}.stderr"
        if ! "${cmd[@]}" > "$per_file" 2> "$per_stderr"; then
            log_warn "  parallel=$np failed"
            continue
        fi
        # Verify the file has at least one parseable JSON line; an empty
        # file (n_kv_max < np*tg+pp) means the test was silently dropped
        # by llama-batched-bench.
        if [[ ! -s "$per_file" ]]; then
            log_warn "  parallel=$np produced no output (likely ctx=$run_ctx too small for np=$np)"
            continue
        fi
        all_results+=("$per_file")
    done

    python3 - "$out_dir" "$model_name" "$backend" "$BATCHED_PROMPT_SIZE" "$BATCHED_GEN_SIZE" "${all_results[@]}" << 'PYEOF'
import json, os, sys
out_dir, model_name, backend = sys.argv[1], sys.argv[2], sys.argv[3]
batched_prompt_size = int(sys.argv[4])
batched_gen_size = int(sys.argv[5])
result_files = sys.argv[6:]

# llama-batched-bench with --output-format jsonl emits one JSON object per
# line. Some llama.cpp versions print a single trailing summary line; we
# accept both shapes and normalise to a list of {n_parallel, tps} dicts.
rows = []
for rf in result_files:
    try:
        for line in open(rf):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict): rows.append(obj)
            elif isinstance(obj, list): rows.extend(obj)
    except Exception as e:
        print(f'[WARN] could not parse {rf}: {e}', file=sys.stderr)

def get_parallel(r):
    for k in ('pl', 'n_parallel', 'parallel', 'n_parallel_seq', 'npl', 'B'):
        if k in r: return r[k]
    return 0

def get_tps(r):
    # llama-batched-bench jsonl rows have:
    #   - 'speed' (total tokens/sec for the batch)
    #   - 'speed_pp' (prompt processing speed) and 'speed_tg' (decode speed)
    for k in ('speed', 's_tps', 'S t/s', 'avg_tps', 'total_tps', 'tps'):
        if k in r: return r[k]
    return 0

def get_pp_tps(r):
    for k in ('speed_pp', 's_pp_tps', 'S_PP t/s', 'pp_tps'):
        if k in r: return r[k]
    return 0

def get_tg_tps(r):
    for k in ('speed_tg', 's_tg_tps', 'S_TG t/s', 'tg_tps'):
        if k in r: return r[k]
    return 0

rows.sort(key=lambda r: get_parallel(r))

with open(os.path.join(out_dir, 'raw.json'), 'w') as f:
    json.dump(rows, f, indent=2)

summary = {
    'parallel': {},
    'pp_per_slot_tps': {},
    'tg_per_slot_tps': {},
    'max_parallel_tps': 0,
}
for r in rows:
    np = get_parallel(r)
    tps = get_tps(r)
    summary['parallel'][str(np)] = tps
    if np == 1:
        summary['parallel_1_tps'] = tps
        summary['pp_1_tps'] = get_pp_tps(r)
        summary['tg_1_tps'] = get_tg_tps(r)
    pp_per = get_pp_tps(r)
    tg_per = get_tg_tps(r)
    if pp_per: summary['pp_per_slot_tps'][str(np)] = pp_per
    if tg_per: summary['tg_per_slot_tps'][str(np)] = tg_per
    if tps > summary['max_parallel_tps']:
        summary['max_parallel_tps'] = tps
        summary['max_parallel'] = np
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

def fmt(x, suf=''):
    try: return f'{float(x):.1f}{suf}'
    except (TypeError, ValueError): return '-'

md = [f'# {model_name} -- llama-batched-bench ({backend})', '']
md.append(f'**Per-test prompt:** {batched_prompt_size} tokens prefill, {batched_gen_size} tokens generate')
md.append('')
md.append('## Concurrent Throughput')
md.append('')
md.append('| Parallel | Total t/s | PP t/s | TG t/s | Per-slot total t/s |')
md.append('|---------:|----------:|-------:|-------:|-------------------:|')
for r in rows:
    np = get_parallel(r)
    total = get_tps(r)
    pp = get_pp_tps(r)
    tg = get_tg_tps(r)
    per_slot = total / np if np else 0
    md.append(f'| {np} | {fmt(total)} | {fmt(pp)} | {fmt(tg)} | {fmt(per_slot)} |')
md.append('')
md.append('_Each row aggregates `n_parallel` simultaneous prompts through the same KV cache._')
md.append(f'_Source: [`raw.json`](./raw.json)_')
md.append('')
with open(os.path.join(out_dir, 'report.md'), 'w') as f:
    f.write('\n'.join(md))
PYEOF

    log_ok "batched complete"
}

# =============================================================================
# Per-model summary.md (combines cache + bench + batched if present)
# =============================================================================

render_model_summary() {
    local backend="$1" model="$2"
    local model_dir_path
    model_dir_path=$(model_dir "$backend" "$model")

    python3 - "$model_dir_path" "$model" "$backend" << 'PYEOF'
import json, os, sys
md_path = sys.argv[1]
model_name = sys.argv[2]
backend = sys.argv[3]

cache_dir = os.path.join(md_path, 'cache')
bench_dir = os.path.join(md_path, 'bench')
batched_dir = os.path.join(md_path, 'batched')

cache_summary = json.load(open(os.path.join(cache_dir, 'summary.json'))) if os.path.exists(os.path.join(cache_dir, 'summary.json')) else None
bench_summary = json.load(open(os.path.join(bench_dir, 'summary.json'))) if os.path.exists(os.path.join(bench_dir, 'summary.json')) else None
batched_summary = json.load(open(os.path.join(batched_dir, 'summary.json'))) if os.path.exists(os.path.join(batched_dir, 'summary.json')) else None

md = [f'# {model_name}', '']
md.append(f'**Backend:** {backend}')
md.append('')

if cache_summary and cache_summary.get('results'):
    p = cache_summary['results'][0].get('profile', {})
    profile = p.get('profile', 'unknown')
    features = p.get('features', [])
    if features:
        md.append(f'**Profile:** {profile} (features: {", ".join(features)})')
    else:
        md.append(f'**Profile:** {profile}')
    md.append('')

if cache_summary:
    md.append('## Cache Performance')
    md.append('')
    md.append('[Drilldown](./cache/analysis.md)')
    md.append('')
    md.append('| Size | Cold TTFT | Warm TTFT | TTFT Speedup | MoE Hit | Cache |')
    md.append('|------|----------:|----------:|-------------:|---------|-------|')
    for r in cache_summary['results']:
        if r.get('error'):
            md.append(f'| {r.get("size_label","?")} | - | - | - | - | error |')
            continue
        cw = r.get('cache_warm', {})
        moe = r.get('moe_warm', {})
        moe_hit = f"{moe.get('hit_rate_pct', 0):.1f}%" if moe.get('present') else '-'
        md.append(
            f'| {r["size_label"]} | {r.get("cold_ttft_ms", 0)}ms | {r.get("warm_ttft_ms", 0)}ms | '
            f'{r.get("prompt_eval_speedup", 0)}x | {moe_hit} | {cw.get("cache_state", "?")} |'
        )
    md.append('')

if bench_summary:
    md.append('## llama-bench')
    md.append('')
    md.append('[Drilldown](./bench/report.md)')
    md.append('')
    md.append('### Prompt Processing (tg=128)')
    md.append('')
    md.append('| pp tokens | pp t/s |')
    md.append('|----------:|-------:|')
    for n in sorted(bench_summary.get('pp_tps', {}), key=lambda x: int(x)):
        md.append(f'| {n} | {bench_summary["pp_tps"][n]:.1f} |')
    md.append('')
    md.append('### Text Generation (pp=512)')
    md.append('')
    md.append('| tg tokens | tg t/s |')
    md.append('|----------:|-------:|')
    for n in sorted(bench_summary.get('tg_tps', {}), key=lambda x: int(x)):
        md.append(f'| {n} | {bench_summary["tg_tps"][n]:.1f} |')
    md.append('')

if batched_summary and batched_summary.get('parallel'):
    md.append('## llama-batched-bench')
    md.append('')
    md.append('[Drilldown](./batched/report.md)')
    md.append('')
    md.append('| Parallel prompts | Total t/s |')
    md.append('|-----------------:|----------:|')
    for n in sorted(batched_summary['parallel'], key=lambda x: int(x)):
        md.append(f'| {n} | {batched_summary["parallel"][n]:.1f} |')
    md.append('')

if not (cache_summary or bench_summary or batched_summary):
    md.append('_No tests completed successfully._')
    md.append('')

md.append('## Files')
md.append('')
if cache_summary: md.append('- [`cache/analysis.md`](./cache/analysis.md) -- cache drilldown')
if bench_summary: md.append('- [`bench/report.md`](./bench/report.md) -- llama-bench drilldown')
if batched_summary: md.append('- [`batched/report.md`](./batched/report.md) -- batched drilldown')
md.append('- [`summary.json`](./summary.json) -- machine-readable aggregate')

with open(os.path.join(md_path, 'summary.md'), 'w') as f:
    f.write('\n'.join(md))

out = {
    'model': model_name,
    'backend': backend,
    'cache': cache_summary,
    'bench': bench_summary,
    'batched': batched_summary,
}
with open(os.path.join(md_path, 'summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
PYEOF
}

# =============================================================================
# Aggregate: per-backend index.md + summary.json
# =============================================================================

render_backend_index() {
    local backend="$1"
    local backend_dir="$BENCH_ROOT/$backend"

    python3 - "$backend_dir" "$backend" << 'PYEOF'
import json, os, sys, glob
backend_dir, backend = sys.argv[1], sys.argv[2]

models = []
for sj in sorted(glob.glob(os.path.join(backend_dir, '*', 'summary.json'))):
    try:
        models.append(json.load(open(sj)))
    except Exception as e:
        print(f'[WARN] could not parse {sj}: {e}', file=sys.stderr)

models.sort(key=lambda m: (m.get('bench') or {}).get('tg_128_tps', 0), reverse=True)

def fmt(x, suf=''):
    try: return f'{float(x):.1f}{suf}'
    except (TypeError, ValueError): return '-'

md = [f'# {backend.upper()} Benchmark Leaderboard', '']
md.append(f'**Models tested:** {len(models)}')
md.append('')

md.append('## Performance at a Glance')
md.append('')
md.append('| Model | tg t/s (128) | tg t/s (512) | pp t/s (2k) | pp t/s (8k) | warm 15k TTFT | cache speedup |')
md.append('|-------|------------:|------------:|------------:|------------:|--------------:|--------------:|')
for m in models:
    bench = m.get('bench') or {}
    cache_results = (m.get('cache') or {}).get('results') or []
    warm_15k = '-'
    speedup_15k = '-'
    for r in cache_results:
        if r.get('size_label') == 'large':
            warm_15k = f"{r.get('warm_ttft_ms', 0):.0f}ms"
            speedup_15k = f"{r.get('prompt_eval_speedup', 0):.1f}x"
            break
    md.append(
        f'| [{m["model"]}]({m["model"]}/summary.md) | '
        f'{fmt(bench.get("tg_128_tps", 0))} | '
        f'{fmt(bench.get("tg_512_tps", 0))} | '
        f'{fmt(bench.get("pp_2048_tps", 0))} | '
        f'{fmt(bench.get("pp_8192_tps", 0))} | '
        f'{warm_15k} | {speedup_15k} |'
    )
md.append('')
md.append('**tg t/s (N)** = text generation throughput with N tokens generated (llama-bench, pp=512).  ')
md.append('**pp t/s (N)** = prompt processing throughput with N-token prefill (llama-bench, tg=128).  ')
md.append('**warm 15k TTFT** = server-side prompt eval after restoring SSD cache for a 15K-token prompt.  ')
md.append('**cache speedup** = cold TTFT / warm TTFT at 15K.')
md.append('')

with open(os.path.join(backend_dir, 'summary.json'), 'w') as f:
    json.dump({'backend': backend, 'models': models}, f, indent=2)

with open(os.path.join(backend_dir, 'index.md'), 'w') as f:
    f.write('\n'.join(md))
PYEOF

    log_ok "Index: $backend_dir/index.md"
}

# =============================================================================
# Per-model orchestration
# =============================================================================

run_one_model() {
    local model="$1" backend="$2"
    local model_name
    model_name=$(basename "$model" .gguf)
    local md
    md=$(model_dir "$backend" "$model_name")

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $model_name"
    echo "  Backend: $backend"
    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    mkdir -p "$md/cache" "$md/bench" "$md/batched"

    local t0=$SECONDS
    local failed=()

    for test in "${SELECTED_TESTS[@]}"; do
        case "$test" in
            cache)
                run_cache_test "$model" "" "$backend" "$md/cache" || failed+=("cache")
                ;;
            bench)
                run_bench_test "$model" "$backend" "$md/bench" || failed+=("bench")
                ;;
            batched)
                run_batched_test "$model" "$backend" "$md/batched" || failed+=("batched")
                ;;
        esac
    done

    render_model_summary "$backend" "$model_name"

    local elapsed=$(( SECONDS - t0 ))
    if [[ ${#failed[@]} -eq 0 ]]; then
        log_ok "$model_name complete in ${elapsed}s"
    else
        log_warn "$model_name done in ${elapsed}s (failed: ${failed[*]})"
    fi
    echo "  Output: $md/"
}

# =============================================================================
# Main
# =============================================================================

export PROJECT_ROOT

trap 'stop_server' EXIT

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

if [[ "$TESTS" == *"cache"* ]]; then
    fetch_source_text || exit 1
fi

for be in rocm vulkan; do
    [[ "$BACKEND" == "both" || "$BACKEND" == "$be" ]] || continue

    mkdir -p "$BENCH_ROOT/$be"

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $be backend"
    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    for model_entry in "${MODELS[@]}"; do
        IFS=':' read -r model extra_flags <<< "$model_entry"
        if [[ ! -f "$MODEL_DIR/$model" ]]; then
            log_warn "Skipping $model (not found)"
            continue
        fi
        run_one_model "$model" "$be"
    done

    render_backend_index "$be"
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Benchmark Complete"
echo "  Results: $BENCH_ROOT/<backend>/index.md"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"