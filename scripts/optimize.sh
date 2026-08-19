#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Optimistic-First Solver for llama.cpp Configuration
# =============================================================================
# Source this file (do not execute directly):
#   source scripts/optimize.sh
#
# Agentic development requires a large context window; the solver will not
# reduce context below MIN_CTX (default 65536) unless absolutely impossible.
# Outputs are exposed via globals prefixed with SOLVER_*. llama-run.sh reads
# these after calling solve_optimal_config() and applies user overrides last.

[[ -n "${_LLAMA_OPTIMIZE_LOADED:-}" ]] && return 0
_LLAMA_OPTIMIZE_LOADED=1

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_OPT_GIB=1073741824

# -----------------------------------------------------------------------------
# Minimum context size for agentic workloads (override via MIN_CTX env)
# -----------------------------------------------------------------------------
: "${MIN_CTX:=65536}"

# -----------------------------------------------------------------------------
# GGUF metadata reader
# -----------------------------------------------------------------------------
declare -A SOLVER_GGUF=()
_SOLVER_GGUF_READER=""

_opt_resolve_gguf_reader() {
    [[ -n "$_SOLVER_GGUF_READER" ]] && return 0
    local candidate
    for candidate in \
        "${PROJECT_ROOT:-.}/scripts/read_gguf_kv.py" \
        "${PROJECT_ROOT:-.}/scratch/read_gguf_kv.py" \
        "./scripts/read_gguf_kv.py" \
        "./scratch/read_gguf_kv.py" \
        "${SCRIPT_DIR:-.}/scratch/read_gguf_kv.py"; do
        if [[ -f "$candidate" ]]; then
            _SOLVER_GGUF_READER="$candidate"
            return 0
        fi
    done
    return 1
}

_opt_read_gguf_meta() {
    local gguf_path="$1"
    SOLVER_GGUF=()
    [[ ! -f "$gguf_path" ]] && return 1
    _opt_resolve_gguf_reader || return 1

    local helper
    helper=$(mktemp /tmp/llama-opt-XXXXXX.py)
    cat > "$helper" <<'PYEOF'
import json, sys, subprocess
try:
    out = subprocess.check_output(
        ["python3", sys.argv[2], sys.argv[1]],
        stderr=subprocess.DEVNULL, text=True, timeout=30)
    d = json.loads(out)
    for k, v in d.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            sys.stdout.write(f"{k}={v}\n")
            short = k.rsplit(".", 1)[-1]
            sys.stdout.write(f"{short}={v}\n")
except Exception:
    sys.exit(1)
PYEOF

    local pairs
    pairs=$(python3 "$helper" "$gguf_path" "$_SOLVER_GGUF_READER" 2>/dev/null)
    local rc=$?
    rm -f "$helper"
    [[ $rc -ne 0 || -z "$pairs" ]] && return 1

    while IFS='=' read -r key val; do
        [[ -z "$key" ]] && continue
        SOLVER_GGUF["$key"]="$val"
    done <<< "$pairs"
    return 0
}

_opt_gguf() {
    local key="$1" default="${2:-0}"
    local val="${SOLVER_GGUF[$key]:-$default}"
    [[ "$val" =~ ^[0-9]+$ ]] && echo "$val" || echo "$default"
}

# -----------------------------------------------------------------------------
# System memory detection (mirrors llama-run.sh functions)
# -----------------------------------------------------------------------------
_opt_get_total_memory_bytes() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n hw.memsize 2>/dev/null || echo 0
    else
        awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

_opt_get_available_memory_bytes() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        local ps=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
        local out=$(vm_stat 2>/dev/null)
        local free=$(echo "$out" | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
        local inactive=$(echo "$out" | awk '/Pages inactive/ {gsub(/\./,"",$3); print $3}')
        echo $(( (${free:-0} + ${inactive:-0}) * ps ))
    else
        awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
    fi
}

# -----------------------------------------------------------------------------
# Memory math
# -----------------------------------------------------------------------------
_opt_kv_type_size() {
    case "$1" in
        f16|bf16) echo 2 ;;
        f32) echo 4 ;;
        q8_0|q4_1|q5_1) echo 1 ;;
        q4_0|q5_0|iq4_nl) echo "0.5625" ;;
        *) echo 2 ;;
    esac
}

_opt_layer_kv_bytes_per_token() {
    local k_type="$1" v_type="$2" head_count_kv="$3" key_len="$4" val_len="$5"
    local k_size v_size
    k_size=$(_opt_kv_type_size "$k_type")
    v_size=$(_opt_kv_type_size "$v_type")
    awk -v k="$k_size" -v v="$v_size" -v h="$head_count_kv" -v kl="$key_len" -v vl="$val_len" \
        'BEGIN { printf "%.0f", h * (kl * k + vl * v) }'
}

_opt_attn_layers() {
    local n_layer="$1" interval="$2"
    interval=${interval:-1}
    if [[ "$interval" -le 1 ]]; then
        echo "$n_layer"
    else
        echo $(( (n_layer + interval - 1) / interval ))
    fi
}

# -----------------------------------------------------------------------------
# Model GPU footprint estimation
# -----------------------------------------------------------------------------
_opt_model_gpu_footprint() {
    local ngl="$1" n_layer="$2" total_model_bytes="$3" moe_strategy="${4:-gpu}" is_ssm_flag="${5:-false}"
    if [[ "$ngl" -le 0 ]]; then
        echo 0
        return
    fi
    local gpu_fraction=1
    case "$moe_strategy" in
        gpu)
            gpu_fraction=1
            ;;
        residency)
            # 30% of model stays GPU-side (attention + small expert cache).
            # This is a conservative estimate for initial load and command
            # submission overhead. Only applies to MoE models - residency has
            # no meaning for dense/SSM models where there are no experts to
            # evict and the full model stays GPU-side. Forcing gpu_fraction=1
            # for non-MoE prevents the solver from picking configs that
            # fit the budget under residency but OOM at runtime.
            if [[ "${is_moe:-false}" == "true" ]]; then
                gpu_fraction=0.30
            else
                gpu_fraction=1
            fi
            ;;
        cpu)
            # --cpu-moe + --load-mode none: only attn/emb/head on GPU (~6%)
            if [[ "${is_moe:-false}" == "true" ]]; then
                gpu_fraction=0.06
            else
                gpu_fraction=1
            fi
            ;;
    esac

    if [[ "$ngl" -ge "$n_layer" ]]; then
        awk -v frac="$gpu_fraction" -v total="$total_model_bytes" \
            'BEGIN { printf "%.0f", frac * total }'
    else
        awk -v ngl="$ngl" -v nl="$n_layer" -v frac="$gpu_fraction" -v total="$total_model_bytes" \
            'BEGIN { printf "%.0f", (ngl / nl) * frac * total }'
    fi
}

# -----------------------------------------------------------------------------
# Memory check helpers (now include MTP overhead)
# -----------------------------------------------------------------------------
_opt_mtp_overhead_bytes() {
    local model_bytes="$1"
    local mtp_bytes=$(( model_bytes * 5 / 100 ))
    [[ $mtp_bytes -lt $(( 512 * 1048576 )) ]] && mtp_bytes=$(( 512 * 1048576 ))
    [[ $mtp_bytes -gt $(( 2048 * 1048576 )) ]] && mtp_bytes=$(( 2048 * 1048576 ))
    echo "$mtp_bytes"
}

_opt_gpu_memory() {
    local offloaded_bytes="$1"
    local kv_per_token="$2"
    local ctx_size="$3"
    local draft_bytes="$4"
    local draft_ctx="$5"
    local n_parallel="$6"
    local ubatch="$7"
    local moe_in_ram="${8:-0}"

    local kv_total=$(( ctx_size * kv_per_token * n_parallel ))
    local draft_kv_total=0
    if [[ $draft_ctx -gt 0 && $draft_bytes -gt 0 ]]; then
        draft_kv_total=$(( draft_ctx * kv_per_token / 4 * n_parallel ))
        local draft_kv_cap=$(( draft_bytes * 10 / 100 ))
        [[ $draft_kv_total -gt $draft_kv_cap ]] && draft_kv_total=$draft_kv_cap
    fi
    local compute_bytes=$(( 512 * 1048576 + ubatch * 256 ))
    local system_bytes=$(( 256 * 1048576 ))

    local mtp_overhead=0
    if [[ "${is_mtp:-false}" == "true" && "${SOLVER_DRAFT_ENABLE:-true}" == "true" ]]; then
        mtp_overhead=$(_opt_mtp_overhead_bytes "${MODEL_BYTES:-0}")
    fi

    local raw=$(( offloaded_bytes + draft_bytes + kv_total + draft_kv_total + compute_bytes + system_bytes + moe_in_ram + mtp_overhead ))
    echo $(( raw * 105 / 100 ))
}

_opt_system_memory() {
    local offloaded_bytes="$1"
    local model_bytes="$2"
    local kv_per_token="$3"
    local ctx_size="$4"
    local draft_bytes="$5"
    local draft_ctx="$6"
    local n_parallel="$7"
    local ubatch="$8"
    local moe_in_ram="${9:-0}"
    local strategy="${10:-gpu}"
    local load_mode="${11:-dio}"
    local ssd_hot_mib="${12:-0}"
    local ssd_warm_mib="${13:-0}"

    local kv_total=$(( ctx_size * kv_per_token * n_parallel ))
    local draft_kv_total=0
    if [[ $draft_ctx -gt 0 && $draft_bytes -gt 0 ]]; then
        draft_kv_total=$(( draft_ctx * kv_per_token / 4 * n_parallel ))
        local draft_kv_cap=$(( draft_bytes * 10 / 100 ))
        [[ $draft_kv_total -gt $draft_kv_cap ]] && draft_kv_total=$draft_kv_cap
    fi

    local ssd_bytes=$(( (ssd_hot_mib + ssd_warm_mib) * 1048576 ))
    local compute_bytes=$(( 512 * 1048576 + ubatch * 256 ))
    local system_bytes=$(( 256 * 1048576 ))

    local model_sys_bytes=0
    case "$strategy" in
        gpu)
            model_sys_bytes=0
            ;;
        residency)
            model_sys_bytes=$(awk -v m="$model_bytes" 'BEGIN { printf "%.0f", m * 0.30 }')
            ;;
        cpu)
            model_sys_bytes="$model_bytes"
            ;;
    esac

    local model_and_offloaded=$model_sys_bytes
    [[ "$strategy" == "gpu" ]] && model_and_offloaded=$(( model_sys_bytes + offloaded_bytes ))
    local raw=$(( model_and_offloaded + draft_bytes + kv_total + draft_kv_total + ssd_bytes + compute_bytes + system_bytes + moe_in_ram ))
    echo $(( raw * 110 / 100 ))
}

# -----------------------------------------------------------------------------
# Cache RAM target helpers
# -----------------------------------------------------------------------------
_opt_min_cache_ram_mib() {
    local model_bytes="$1"
    local size_gb=$(( model_bytes / _OPT_GIB ))
    local low_vram=0
    [[ ${SOLVER_GPU_BUDGET_BYTES:-0} -gt 0 && ${SOLVER_GPU_BUDGET_BYTES} -lt $(( 32 * _OPT_GIB )) ]] && low_vram=1

    if [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" && "${is_moe:-false}" == "true" && $size_gb -ge 50 ]]; then
        echo 4096  # was 8192, lowered to allow more combos
    elif [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" && "${is_moe:-false}" == "true" ]]; then
        echo 3072
    elif [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]]; then
        echo 2048
    elif [[ $low_vram -eq 1 && "${is_moe:-false}" == "true" ]]; then
        echo 1024
    else
        echo 1024
    fi
}

# Set SOLVER_CACHE_RAM (prompt cache) and SOLVER_SSD_HOT_RAM/WARM_RAM (SSD cache)
# based on actual leftover system memory after accounting for model, KV, draft, etc.
_opt_update_cache_ram() {
    local solver_budget_bytes="$1"
    local offloaded_bytes="$2"
    local kv_per_token="$3"
    local ctx_size="$4"
    local draft_bytes="$5"
    local n_parallel="$6"
    local ubatch="$7"

    [[ $solver_budget_bytes -le 0 ]] && return 0

    # GPU leftover
    local used_gpu
    used_gpu=$(_opt_gpu_memory \
        "$offloaded_bytes" \
        "$kv_per_token" \
        "$ctx_size" \
        "$draft_bytes" \
        "$ctx_size" \
        "$n_parallel" \
        "$ubatch" \
        0)
    local gpu_leftover=$(( solver_budget_bytes - used_gpu ))

    # System memory leftover (without caches)
    local sys_needed_no_cache
    sys_needed_no_cache=$(_opt_system_memory \
        "$offloaded_bytes" \
        "${MODEL_BYTES:-0}" \
        "$kv_per_token" \
        "$ctx_size" \
        "$draft_bytes" \
        "$ctx_size" \
        "$n_parallel" \
        "$ubatch" \
        0 \
        "${SOLVER_MOE_STRATEGY:-gpu}" \
        "$SOLVER_LOAD_MODE" \
        0 \
        0)

    local sys_total_mib=$(( $(_opt_get_total_memory_bytes) / 1048576 ))
    local os_reserve_mib
    if [[ $sys_total_mib -le 16384 ]]; then
        os_reserve_mib=$(( sys_total_mib / 4 ))
    else
        os_reserve_mib=8192   # 8 GiB for systems >16 GiB
    fi
    local sys_budget_mib=$(( sys_total_mib - os_reserve_mib ))
    local sys_leftover=$(( sys_budget_mib * 1048576 - sys_needed_no_cache ))

    # Effective leftover is the tighter of GPU and system
    local effective_leftover=$(( gpu_leftover < sys_leftover ? gpu_leftover : sys_leftover ))
    [[ $effective_leftover -lt 0 ]] && effective_leftover=0

    # Convert to MiB, apply 10% headroom
    local total_cache_mib=$(( effective_leftover * 90 / 100 / 1048576 ))
    [[ $total_cache_mib -lt 256 ]] && total_cache_mib=256

    # Absolute cap: never exceed 25% of total system RAM for all caches combined
    local max_total_cache_mib=$(( sys_total_mib / 4 ))
    [[ $total_cache_mib -gt $max_total_cache_mib ]] && total_cache_mib=$max_total_cache_mib

    # Split between prompt cache and SSD cache (if enabled)
    local prompt_cache_mib=$total_cache_mib
    local ssd_cache_mib=0
    if [[ "$SOLVER_SSD_ENABLE" == "true" ]]; then
        ssd_cache_mib=$(( total_cache_mib / 5 ))
        [[ $ssd_cache_mib -gt 1024 ]] && ssd_cache_mib=1024
        [[ $ssd_cache_mib -lt 128 ]] && ssd_cache_mib=128
        prompt_cache_mib=$(( total_cache_mib - ssd_cache_mib ))
    fi

    SOLVER_SSD_HOT_RAM=$(( ssd_cache_mib / 2 ))
    SOLVER_SSD_WARM_RAM=$(( ssd_cache_mib - SOLVER_SSD_HOT_RAM ))
    SOLVER_CACHE_RAM=$prompt_cache_mib

    SOLVER_REASONS+=("cache-ram: ${prompt_cache_mib} MiB (SSD: ${ssd_cache_mib} MiB)")
}

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_opt_start_optimistic() {
    local effective_tier="${LLAMA_HARDWARE_TIER:-standard}"
    [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]] && effective_tier="halo"
    SOLVER_TIER="$effective_tier"

    SOLVER_MOE_STRATEGY="gpu"
    SOLVER_LOAD_MODE="dio"

    local ctx_train=$(_opt_gguf context_length 32768)
    local ctx_cap=$(( ctx_train * 4 ))
    [[ $ctx_cap -gt 1048576 ]] && ctx_cap=1048576
    SOLVER_CTX_SIZE="$ctx_cap"
    [[ $SOLVER_CTX_SIZE -lt $MIN_CTX ]] && SOLVER_CTX_SIZE=$MIN_CTX

    SOLVER_K_TYPE="f16"
    SOLVER_V_TYPE="f16"

    case "$effective_tier" in
        halo)      SOLVER_UBATCH=2048; SOLVER_BATCH=4096 ;;
        standard)  SOLVER_UBATCH=1024; SOLVER_BATCH=2048 ;;
        handheld)  SOLVER_UBATCH=512;  SOLVER_BATCH=1024 ;;
        *)         SOLVER_UBATCH=1024; SOLVER_BATCH=2048 ;;
    esac

    local phys_cores=${PHYSICAL_CORES:-}
    if [[ -z "$phys_cores" ]]; then
        if command -v lscpu &>/dev/null; then
            phys_cores=$(lscpu -p 2>/dev/null | grep -E '^[0-9]' | cut -d',' -f2 | sort -u | wc -l)
        else
            phys_cores=$(nproc)
            [[ -f /sys/devices/system/cpu/smt/active ]] && [[ "$(cat /sys/devices/system/cpu/smt/active)" == "1" ]] && phys_cores=$((phys_cores / 2))
        fi
    fi
    [[ -z "$phys_cores" || "$phys_cores" -lt 1 ]] && phys_cores=1

    case "$effective_tier" in
        halo)      SOLVER_THREADS_BATCH="$phys_cores"; SOLVER_THREADS="$(( phys_cores / 2 ))" ;;
        standard)  SOLVER_THREADS_BATCH="$phys_cores"; SOLVER_THREADS="$(( phys_cores / 2 ))" ;;
        handheld)  SOLVER_THREADS_BATCH="$(( phys_cores / 2 ))"; SOLVER_THREADS="$(( phys_cores / 4 ))" ;;
        *)         SOLVER_THREADS_BATCH="$phys_cores"; SOLVER_THREADS="$(( phys_cores / 2 ))" ;;
    esac
    [[ $SOLVER_THREADS_BATCH -lt 1 ]] && SOLVER_THREADS_BATCH=1
    [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1

    local sys_total_mib=$(( $(_opt_get_total_memory_bytes) / 1048576 ))
    if [[ $sys_total_mib -le 32768 ]]; then
        SOLVER_SSD_ENABLE=true
        SOLVER_SSD_HOT_RAM=512
        SOLVER_SSD_WARM_RAM=512
    else
        SOLVER_SSD_ENABLE=true
        [[ "$effective_tier" == "halo" ]] && SOLVER_SSD_ENABLE=false
        case "$effective_tier" in
            halo)      SOLVER_SSD_HOT_RAM=2048; SOLVER_SSD_WARM_RAM=2048 ;;
            standard)  SOLVER_SSD_HOT_RAM=960;  SOLVER_SSD_WARM_RAM=1440 ;;
            handheld)  SOLVER_SSD_HOT_RAM=512;  SOLVER_SSD_WARM_RAM=768  ;;
            *)         SOLVER_SSD_HOT_RAM=960;  SOLVER_SSD_WARM_RAM=1440 ;;
        esac
    fi

    SOLVER_DRAFT_ENABLE=true
    SOLVER_DRAFT_N_MAX=8
    SOLVER_MOE_RESIDENT_PER_LAYER=32
    SOLVER_MOE_PREWARM_TOP_K=16
    SOLVER_LOAD_MODE="${SOLVER_LOAD_MODE:-dio}"
    SOLVER_VK_NPS="${GGML_VK_NODES_PER_SUBMIT:-}"
    SOLVER_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-8192}"

    # Default checkpoint count (auto-scaled at end of solve_optimal_config)
    SOLVER_CHECKPOINTS="${SOLVER_CHECKPOINTS:-}"

    SOLVER_NGL="$SOLVER_N_LAYER"
}

_reduce_ctx() {
    local min_ctx=$MIN_CTX
    [[ $SOLVER_CTX_SIZE -le $min_ctx ]] && return 1
    local ctx_values=(262144 196608 131072 98304 65536)
    for c in "${ctx_values[@]}"; do
        if [[ $SOLVER_CTX_SIZE -gt $c && $c -ge $min_ctx ]]; then
            SOLVER_CTX_SIZE=$c
            SOLVER_REASONS+=("ctx: ${c}")
            return 0
        fi
    done
    return 1
}

_reduce_ngl() {
    [[ "${_SOLVER_DONE_reduce_ngl:-0}" == "1" ]] && return 1
    local total_layers="$SOLVER_N_LAYER"
    local current="$SOLVER_NGL"
    [[ $current -le 0 ]] && { _SOLVER_DONE_reduce_ngl=1; return 1; }
    local reduction=$(( total_layers / 10 ))
    [[ $reduction -lt 1 ]] && reduction=1
    local new_ngl=$(( current - reduction ))
    [[ $new_ngl -lt 0 ]] && new_ngl=0
    SOLVER_NGL="$new_ngl"
    SOLVER_REASONS+=("ngl: $new_ngl")
    [[ $new_ngl -eq 0 ]] && _SOLVER_DONE_reduce_ngl=1
    return 0
}

_reduce_ssd_ram() {
    [[ "$SOLVER_SSD_ENABLE" != "true" ]] && return 1
    [[ $SOLVER_SSD_HOT_RAM -le 128 && $SOLVER_SSD_WARM_RAM -le 128 ]] && return 1
    SOLVER_SSD_HOT_RAM=$(( SOLVER_SSD_HOT_RAM / 2 ))
    [[ $SOLVER_SSD_HOT_RAM -lt 128 ]] && SOLVER_SSD_HOT_RAM=128
    SOLVER_SSD_WARM_RAM=$(( SOLVER_SSD_WARM_RAM / 2 ))
    [[ $SOLVER_SSD_WARM_RAM -lt 128 ]] && SOLVER_SSD_WARM_RAM=128
    SOLVER_REASONS+=("SSD RAM: ${SOLVER_SSD_HOT_RAM}/${SOLVER_SSD_WARM_RAM} MiB")
    return 0
}

_reduce_kv_q8_0() {
    [[ "${_SOLVER_DONE_kv_q8_0:-0}" == "1" ]] && return 1
    SOLVER_K_TYPE="q8_0"
    SOLVER_V_TYPE="q8_0"
    _SOLVER_DONE_kv_q8_0=1
    SOLVER_REASONS+=("KV: q8_0/q8_0")
    return 0
}

_reduce_kv_q4_0() {
    [[ "${_SOLVER_DONE_kv_q4_0:-0}" == "1" ]] && return 1
    SOLVER_K_TYPE="q4_0"
    SOLVER_V_TYPE="q4_0"
    _SOLVER_DONE_kv_q4_0=1
    SOLVER_REASONS+=("KV: q4_0/q4_0")
    return 0
}

_drop_draft() {
    [[ "${_SOLVER_DONE_drop_draft:-0}" == "1" || "$SOLVER_DRAFT_ENABLE" != "true" ]] && return 1
    SOLVER_DRAFT_ENABLE=false
    _SOLVER_DONE_drop_draft=1
    SOLVER_REASONS+=("draft: dropped")
    return 0
}

_drop_ssd() {
    [[ "${_SOLVER_DONE_drop_ssd:-0}" == "1" || "$SOLVER_SSD_ENABLE" != "true" ]] && return 1
    SOLVER_SSD_ENABLE=false
    SOLVER_SSD_HOT_RAM=0
    SOLVER_SSD_WARM_RAM=0
    _SOLVER_DONE_drop_ssd=1
    SOLVER_REASONS+=("SSD: disabled")
    return 0
}

_reduce_ubatch() {
    [[ $SOLVER_UBATCH -le 512 ]] && return 1
    SOLVER_UBATCH=$(( SOLVER_UBATCH / 2 ))
    [[ $SOLVER_BATCH -gt $SOLVER_UBATCH ]] && SOLVER_BATCH=$SOLVER_UBATCH
    SOLVER_REASONS+=("ubatch: ${SOLVER_UBATCH}")
    return 0
}

_opt_detune_steps() {
    cat <<'STEPS'
_reduce_kv_q8_0
_reduce_kv_q4_0
_reduce_ngl
_reduce_ssd_ram
_drop_draft
_drop_ssd
_reduce_ubatch
_reduce_ctx
STEPS
}

solve_optimal_config() {
    local model_path="$1"

    _SOLVER_DONE_drop_draft=0
    _SOLVER_DONE_drop_ssd=0
    _SOLVER_DONE_reduce_ngl=0
    _SOLVER_DONE_reduce_ubatch=0
    _SOLVER_DONE_kv_q8_0=0
    _SOLVER_DONE_kv_q4_0=0

    _opt_read_gguf_meta "$model_path" || SOLVER_GGUF=()

    local n_layer=$(_opt_gguf block_count 32)
    SOLVER_N_LAYER="$n_layer"

    _opt_start_optimistic

    local draft_path=""
    [[ -n "${prof_dspark:-}" ]] && draft_path="$prof_dspark"
    [[ -z "$draft_path" && -n "${prof_dflash:-}" ]] && draft_path="$prof_dflash"
    local draft_bytes=0
    if [[ -n "$draft_path" && -f "$draft_path" ]]; then
        draft_bytes=$(stat -c%s "$draft_path" 2>/dev/null || stat -f%z "$draft_path" 2>/dev/null || echo 0)
    fi

    local fai=$(_opt_gguf full_attention_interval 1)
    local n_attn=$(_opt_attn_layers "$n_layer" "$fai")

    local hckv=$(_opt_gguf head_count_kv 0)
    [[ $hckv -eq 0 ]] && hckv=$(_opt_gguf n_head_kv 0)
    [[ $hckv -eq 0 ]] && hckv=$(_opt_gguf head_count 0)
    [[ $hckv -eq 0 ]] && hckv=$(_opt_gguf n_head 0)
    if [[ $hckv -eq 0 ]]; then
        local size_gb=$(( MODEL_BYTES / 1073741824 ))
        if [[ $size_gb -gt 40 ]]; then
            hckv=8
        elif [[ $size_gb -gt 20 ]]; then
            hckv=4
        else
            hckv=2
        fi
    fi

    local embd=$(_opt_gguf embedding_length 0)
    [[ $embd -eq 0 ]] && embd=$(_opt_gguf n_embd 0)
    local hc=$(_opt_gguf head_count 0)
    [[ $hc -eq 0 ]] && hc=$(_opt_gguf n_head 0)

    local kl=0 vl=0
    kl=$(_opt_gguf key_length 0)
    vl=$(_opt_gguf value_length 0)
    if [[ $kl -eq 0 || $vl -eq 0 ]]; then
        if [[ $embd -gt 0 && $hc -gt 0 ]]; then
            local head_dim=$(( embd / hc ))
            [[ $kl -eq 0 ]] && kl=$head_dim
            [[ $vl -eq 0 ]] && vl=$head_dim
        else
            local size_gb=$(( MODEL_BYTES / 1073741824 ))
            if [[ $size_gb -gt 40 ]]; then
                kl=128; vl=128
            else
                kl=256; vl=256
            fi
        fi
    fi
    [[ $kl -eq 0 ]] && kl=256
    [[ $vl -eq 0 ]] && vl=256

    SOLVER_REASONS=()
    SOLVER_DRAFT_PATH="$draft_path"

    local solver_budget_bytes
    if [[ ${SOLVER_GPU_BUDGET_BYTES:-${GPU_BUDGET_BYTES:-0}} -gt 0 ]]; then
        solver_budget_bytes=${SOLVER_GPU_BUDGET_BYTES:-$GPU_BUDGET_BYTES}
    else
        solver_budget_bytes=0
    fi

    local low_vram=0
    [[ $solver_budget_bytes -gt 0 && $solver_budget_bytes -lt $(( 32 * _OPT_GIB )) ]] && low_vram=1

    # -------------------------------------------------------------------------
    # Strategy selection:
    #   - default: gpu -> residency -> cpu
    #   - We now reorder for MoE when CPU-MoE is viable: gpu -> cpu -> residency
    #     to avoid Vulkan OOM from residency on large models.
    # -------------------------------------------------------------------------
    local strategies=("gpu")
    if [[ "${is_moe:-false}" == "true" ]]; then
        # Check if CPU-MoE is viable: model fits in system RAM with OS reserve.
        local sys_total_bytes=$(_opt_get_total_memory_bytes)
        local os_reserve_bytes=$(( 8 * _OPT_GIB ))
        local sys_avail_bytes=$(( sys_total_bytes - os_reserve_bytes ))
        local model_fits_cpu=0
        [[ ${MODEL_BYTES:-0} -le $sys_avail_bytes ]] && model_fits_cpu=1

        # If model is >80% of GPU budget and CPU-MoE is viable, skip residency
        # and use CPU only. Residency can still OOM on UMA APUs because the
        # driver may map the full model into GPU-accessible memory.
        if [[ $model_fits_cpu -eq 1 && ${MODEL_BYTES:-0} -gt $(( solver_budget_bytes * 80 / 100 )) ]]; then
            strategies=("gpu" "cpu")
        else
            strategies+=("residency" "cpu")
        fi
    fi
    # Dense/SSM models: residency and cpu are meaningless (no experts to
    # evict). Only the gpu strategy is valid, so we leave strategies=("gpu").

    # -------------------------------------------------------------------------
    # Build scored combinations
    # -------------------------------------------------------------------------
    local ctx_values=(262144 196608 131072 98304 65536)
    if [[ $low_vram -eq 1 && "${is_moe:-false}" == "true" ]]; then
        ctx_values=(65536)
    fi

    local kv_qualities=("f16/f16" "q8_0/q8_0" "q4_0/q4_0")
    if [[ $low_vram -eq 1 && "${is_moe:-false}" == "true" ]]; then
        kv_qualities=("q8_0/q8_0" "q4_0/q4_0")
    fi

    declare -A combo_score
    for strategy in "${strategies[@]}"; do
        for ctx in "${ctx_values[@]}"; do
            for kvq in "${kv_qualities[@]}"; do
                local draft_modes_for_strategy=("enabled")
                [[ "$strategy" == "gpu" ]] && draft_modes_for_strategy=("enabled" "disabled")
                for draft_mode in "${draft_modes_for_strategy[@]}"; do
                    local strategy_score=0
                    [[ "$strategy" == "gpu" ]] && strategy_score=300
                    [[ "$strategy" == "cpu" ]] && strategy_score=250
                    [[ "$strategy" == "residency" ]] && strategy_score=200

                    local ctx_score=0
                    case "$ctx" in
                        131072) ctx_score=100 ;;
                        98304)  ctx_score=95  ;;
                        196608) ctx_score=90  ;;
                        262144) ctx_score=85  ;;
                        65536)  ctx_score=80  ;;
                        *)      ctx_score=$(( ctx / 1000 )) ;;
                    esac

                    local kv_score=0
                    [[ "$kvq" == "f16/f16" ]] && kv_score=30
                    [[ "$kvq" == "q8_0/q8_0" ]] && kv_score=20
                    [[ "$kvq" == "q4_0/q4_0" ]] && kv_score=10

                    local draft_score=0
                    [[ "$draft_mode" == "enabled" ]] && draft_score=5

                    combo_score["${strategy}:${ctx}:${kvq}:${draft_mode}"]=$(( strategy_score * 1000 + ctx_score * 10 + kv_score + draft_score ))
                done
            done
        done
    done

    local sorted_combos=()
    for combo in "${!combo_score[@]}"; do
        sorted_combos+=("${combo_score[$combo]}:$combo")
    done
    IFS=$'\n' sorted_combos=($(sort -rn <<< "${sorted_combos[*]}"))
    unset IFS

    local chosen_strategy=""
    local chosen_ctx=0
    local chosen_kvq=""
    local chosen_k_type=""
    local chosen_v_type=""
    local chosen_draft_enable=true
    local found=0

    for scored_combo in "${sorted_combos[@]}"; do
        local score="${scored_combo%%:*}"
        local combo="${scored_combo#*:}"
        local strategy="${combo%%:*}"
        local rest="${combo#*:}"
        local ctx="${rest%%:*}"
        rest="${rest#*:}"
        local kvq="${rest%%:*}"
        local draft_mode="${rest#*:}"

        [[ "$strategy" == "cpu" && $ctx -lt 8192 ]] && continue
        [[ $ctx -lt $MIN_CTX && "$strategy" != "cpu" ]] && continue

        local k_type="${kvq%%/*}"
        local v_type="${kvq##*/}"

        local kv_per_token_per_layer
        kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$k_type" "$v_type" "$hckv" "$kl" "$vl")
        local kv_per_token=$(( kv_per_token_per_layer * n_attn ))

        local eff_draft_bytes=0
        [[ "$draft_mode" == "enabled" && "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"

        local offloaded_bytes
        offloaded_bytes=$(_opt_model_gpu_footprint "$SOLVER_NGL" "$SOLVER_N_LAYER" "${MODEL_BYTES:-0}" "$strategy" "${is_ssm:-false}")

        # System memory check
        local sys_mem_total
        sys_mem_total=$(_opt_get_total_memory_bytes)
        local sys_mem_avail
        sys_mem_avail=$(_opt_get_available_memory_bytes)
        local sys_mem_needed
        sys_mem_needed=$(_opt_system_memory \
            "$offloaded_bytes" \
            "${MODEL_BYTES:-0}" \
            "$kv_per_token" \
            "$ctx" \
            "$eff_draft_bytes" \
            "$ctx" \
            "${OVERRIDE_N_PARALLEL:-1}" \
            "$SOLVER_UBATCH" \
            0 \
            "$strategy" \
            "$SOLVER_LOAD_MODE" \
            "$SOLVER_SSD_HOT_RAM" \
            "$SOLVER_SSD_WARM_RAM")

        local mem_needed
        mem_needed=$(_opt_gpu_memory \
            "$offloaded_bytes" \
            "$kv_per_token" \
            "$ctx" \
            "$eff_draft_bytes" \
            "$ctx" \
            "${OVERRIDE_N_PARALLEL:-1}" \
            "$SOLVER_UBATCH" \
            0)

        local sys_budget
        case "$strategy" in
            residency|cpu)
                local os_reserve=$(( 8 * _OPT_GIB ))
                sys_budget=$(( sys_mem_total - os_reserve ))
                [[ $sys_budget -lt 0 ]] && sys_budget=0
                ;;
            *)
                sys_budget=$sys_mem_avail
                ;;
        esac

        local solver_budget_gib=$(( solver_budget_bytes / _OPT_GIB ))
        local mem_needed_gib=$(( mem_needed / _OPT_GIB ))
        local gpu_ok=0
        [[ $mem_needed_gib -le $solver_budget_gib ]] || [[ $solver_budget_gib -le 0 ]] && gpu_ok=1
        local sys_ok=0
        [[ $sys_mem_needed -le $sys_budget ]] || [[ $sys_budget -le 0 ]] && sys_ok=1

        if [[ $gpu_ok -eq 1 && $sys_ok -eq 1 ]]; then
            local min_cache_mib
            min_cache_mib=$(_opt_min_cache_ram_mib "${MODEL_BYTES:-0}")

            local leftover_mib=0
            [[ $solver_budget_bytes -gt 0 ]] && leftover_mib=$(( (solver_budget_bytes - mem_needed) / 1048576 ))

            if [[ $solver_budget_bytes -gt 0 && $leftover_mib -lt $min_cache_mib ]]; then
                continue
            fi

            chosen_strategy="$strategy"
            chosen_ctx=$ctx
            chosen_kvq="$kvq"
            chosen_k_type="$k_type"
            chosen_v_type="$v_type"
            [[ "$draft_mode" == "enabled" ]] && chosen_draft_enable=true || chosen_draft_enable=false
            found=1
            break
        fi
    done

    if [[ $found -eq 0 ]]; then
        SOLVER_CTX_SIZE=$MIN_CTX
        SOLVER_K_TYPE="q4_0"
        SOLVER_V_TYPE="q4_0"
        SOLVER_REASONS+=("ctx: ${MIN_CTX} KV: q4_0/q4_0 (no fit at any strategy/kv/ctx)")
    else
        SOLVER_CTX_SIZE=$chosen_ctx
        SOLVER_K_TYPE=$chosen_k_type
        SOLVER_V_TYPE=$chosen_v_type
        SOLVER_DRAFT_ENABLE=$chosen_draft_enable
        case "$chosen_strategy" in
            cpu)        SOLVER_MOE_STRATEGY="cpu"; SOLVER_LOAD_MODE="none" ;;
            residency)  SOLVER_MOE_STRATEGY="residency"; SOLVER_LOAD_MODE="mmap" ;;
            *)          SOLVER_MOE_STRATEGY="gpu"; SOLVER_LOAD_MODE="dio" ;;
        esac
        local draft_str=""
        [[ "$chosen_draft_enable" == "true" ]] && draft_str=" draft=on" || draft_str=" draft=off"
        SOLVER_REASONS+=("ctx: ${chosen_ctx} KV: ${chosen_kvq} strategy=${chosen_strategy}${draft_str}")
    fi

    # Recompute kv_per_token for chosen config
    local kv_per_token_per_layer
    kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
    local kv_per_token=$(( kv_per_token_per_layer * n_attn ))

    # Phase 2: fine-tune if still over budget
    local step_idx=0
    while [[ $step_idx -lt 50 ]]; do
        local offloaded_bytes
        offloaded_bytes=$(_opt_model_gpu_footprint "$SOLVER_NGL" "$SOLVER_N_LAYER" "${MODEL_BYTES:-0}" "${SOLVER_MOE_STRATEGY:-gpu}" "${is_ssm:-false}")

        local eff_draft_bytes=0
        [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"

        local mem_needed
        mem_needed=$(_opt_gpu_memory \
            "$offloaded_bytes" \
            "$kv_per_token" \
            "$SOLVER_CTX_SIZE" \
            "$eff_draft_bytes" \
            "$SOLVER_CTX_SIZE" \
            "${OVERRIDE_N_PARALLEL:-1}" \
            "$SOLVER_UBATCH" \
            0)

        local sys_mem_total
        sys_mem_total=$(_opt_get_total_memory_bytes)
        local sys_mem_avail
        sys_mem_avail=$(_opt_get_available_memory_bytes)
        local sys_mem_needed
        sys_mem_needed=$(_opt_system_memory \
            "$offloaded_bytes" \
            "${MODEL_BYTES:-0}" \
            "$kv_per_token" \
            "$SOLVER_CTX_SIZE" \
            "$eff_draft_bytes" \
            "$SOLVER_CTX_SIZE" \
            "${OVERRIDE_N_PARALLEL:-1}" \
            "$SOLVER_UBATCH" \
            0 \
            "${SOLVER_MOE_STRATEGY:-gpu}" \
            "$SOLVER_LOAD_MODE" \
            "$SOLVER_SSD_HOT_RAM" \
            "$SOLVER_SSD_WARM_RAM")

        local sys_budget
        case "${SOLVER_MOE_STRATEGY:-gpu}" in
            residency|cpu)
                local os_reserve=$(( 8 * _OPT_GIB ))
                sys_budget=$(( sys_mem_total - os_reserve ))
                [[ $sys_budget -lt 0 ]] && sys_budget=0
                ;;
            *)
                sys_budget=$sys_mem_avail
                ;;
        esac

        local solver_budget_gib=$(( solver_budget_bytes / _OPT_GIB ))
        local mem_needed_gib=$(( mem_needed / _OPT_GIB ))
        if [[ $mem_needed_gib -le $solver_budget_gib ]] || [[ $solver_budget_gib -le 0 ]]; then
            local sys_ok=0
            [[ $sys_mem_needed -le $sys_budget ]] || [[ $sys_budget -le 0 ]] && sys_ok=1
            [[ $sys_ok -eq 1 ]] && break
        fi

        local applied=0
        local step_fn
        while IFS= read -r step_fn; do
            [[ -z "$step_fn" || "$step_fn" == \#* ]] && continue
            if "$step_fn" 2>/dev/null; then
                applied=1
                break
            fi
        done < <(_opt_detune_steps_phase2)

        [[ $applied -eq 0 ]] && break
        step_idx=$(( step_idx + 1 ))
    done

    # Final cache RAM derivation
    kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
    kv_per_token=$(( kv_per_token_per_layer * n_attn ))

    local final_offloaded
    final_offloaded=$(_opt_model_gpu_footprint \
        "$SOLVER_NGL" \
        "$SOLVER_N_LAYER" \
        "${MODEL_BYTES:-0}" \
        "${SOLVER_MOE_STRATEGY:-gpu}" \
        "${is_ssm:-false}")

    local final_draft_bytes=0
    [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && final_draft_bytes="$draft_bytes"

    _opt_update_cache_ram \
        "$solver_budget_bytes" \
        "$final_offloaded" \
        "$kv_per_token" \
        "$SOLVER_CTX_SIZE" \
        "$final_draft_bytes" \
        "${OVERRIDE_N_PARALLEL:-1}" \
        "$SOLVER_UBATCH"

    local kv_cache_bytes=$(( SOLVER_CTX_SIZE * kv_per_token ))
    SOLVER_KV_CACHE_MIB=$(( kv_cache_bytes / 1048576 + 1024 ))

    SOLVER_PROFILE_NAME=$(_opt_pick_legacy_profile "$n_attn" "${MODEL_BYTES:-0}")

    # P2: Auto-scale checkpoint count based on context size
    # Base: 8 checkpoints at 65K context
    # Scale: +1 checkpoint per 8K context above 65K
    # Cap: 32 checkpoints max
    if [[ -z "${SOLVER_CHECKPOINTS:-}" ]]; then
        local base_ctx=65536
        local base_cp=8
        local scale_per=8192
        local max_cp=32
        if [[ $SOLVER_CTX_SIZE -gt $base_ctx ]]; then
            local extra=$(( (SOLVER_CTX_SIZE - base_ctx) / scale_per ))
            SOLVER_CHECKPOINTS=$(( base_cp + extra ))
        else
            SOLVER_CHECKPOINTS=$base_cp
        fi
        [[ $SOLVER_CHECKPOINTS -gt $max_cp ]] && SOLVER_CHECKPOINTS=$max_cp
        SOLVER_REASONS+=("checkpoints: ${SOLVER_CHECKPOINTS}")
    fi
}

_opt_detune_steps_phase2() {
    cat <<'STEPS'
_reduce_kv_q8_0
_reduce_kv_q4_0
_reduce_ngl
_reduce_ssd_ram
_drop_draft
_drop_ssd
_reduce_ubatch
STEPS
}

_opt_pick_legacy_profile() {
    local n_attn=$1 model_bytes=$2
    local is_strix_halo=false
    [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]] && is_strix_halo=true
    local size_gb=$(( model_bytes / 1073741824 ))

    if [[ "${is_ssm:-false}" == "true" ]]; then
        echo "ssm"
    elif $is_strix_halo && [[ "${is_moe:-false}" == "true" ]]; then
        [[ $size_gb -gt 50 ]] && echo "halo-moe-large" || echo "halo-moe-small"
    elif $is_strix_halo; then
        echo "halo-dense"
    elif [[ "${is_moe:-false}" == "true" ]]; then
        [[ $size_gb -gt 18 ]] && echo "std-moe-large" || echo "std-moe-small"
    else
        [[ $size_gb -gt 15 ]] && echo "std-dense-large" || echo "std-dense"
    fi
}

apply_user_overrides() {
    SOLVER_OVERRIDES=()
    [[ -n "${USER_CTX_SIZE:-}" ]] && { SOLVER_CTX_SIZE="$CTX_SIZE"; SOLVER_OVERRIDES+=("ctx-size"); }
    [[ -n "${USER_KV_CACHE_TYPE:-}" ]] && { SOLVER_K_TYPE="$KV_CACHE_TYPE_K"; SOLVER_V_TYPE="$KV_CACHE_TYPE_V"; SOLVER_OVERRIDES+=("kv-cache-type"); }
    [[ -n "${KV_CACHE_K_OVERRIDE:-}" ]] && { SOLVER_K_TYPE="$KV_CACHE_K_OVERRIDE"; SOLVER_OVERRIDES+=("cache-type-k"); }
    [[ -n "${KV_CACHE_V_OVERRIDE:-}" ]] && { SOLVER_V_TYPE="$KV_CACHE_V_OVERRIDE"; SOLVER_OVERRIDES+=("cache-type-v"); }

    if [[ -n "${LLAMA_THREADS_OVERRIDE:-}" ]]; then
        SOLVER_THREADS_BATCH="$LLAMA_THREADS_OVERRIDE"
        SOLVER_THREADS="$(( LLAMA_THREADS_OVERRIDE / 2 ))"
        [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1
        SOLVER_OVERRIDES+=("threads")
    elif [[ -n "${LLAMA_THREADS:-}" ]]; then
        SOLVER_THREADS_BATCH="$LLAMA_THREADS"
        SOLVER_THREADS="$(( LLAMA_THREADS / 2 ))"
        [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1
    fi

    [[ -n "${MOE_UBATCH_OVERRIDE:-}" ]] && { SOLVER_UBATCH="$MOE_UBATCH_OVERRIDE"; SOLVER_OVERRIDES+=("ubatch"); }
    [[ -n "${OVERRIDE_UBATCH_SIZE:-}" ]] && { SOLVER_UBATCH="$OVERRIDE_UBATCH_SIZE"; SOLVER_OVERRIDES+=("ubatch-size"); }
    [[ -n "${OVERRIDE_CACHE_RAM:-}" ]] && { SOLVER_CACHE_RAM="$OVERRIDE_CACHE_RAM"; SOLVER_SSD_HOT_RAM=0; SOLVER_SSD_WARM_RAM=0; SOLVER_OVERRIDES+=("cache-ram"); }
    if [[ "${_SSD_DISABLE:-false}" == "true" ]]; then
        SOLVER_SSD_ENABLE=false
        SOLVER_OVERRIDES+=("no-ssd-cache")
    fi
    if [[ -n "${OVERRIDE_NGL:-}" ]]; then
        SOLVER_NGL="$OVERRIDE_NGL"
        SOLVER_OVERRIDES+=("ngl-override")
    fi
    [[ "${OVERRIDE_FIT:-}" == "on" ]] && { SOLVER_NGL=-1; SOLVER_OVERRIDES+=("--fit on"); }
    [[ -n "${OVERRIDE_REASONING_BUDGET:-}" ]] && { SOLVER_REASONING_BUDGET="$OVERRIDE_REASONING_BUDGET"; SOLVER_OVERRIDES+=("reasoning-budget"); }
}
