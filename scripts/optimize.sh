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
# Minimum context size for agentic workloads (override via MIN_CTX env)
# -----------------------------------------------------------------------------
: "${MIN_CTX:=65536}"

# -----------------------------------------------------------------------------
# GGUF metadata reader
# -----------------------------------------------------------------------------
declare -A SOLVER_GGUF=()

# Path to the python GGUF reader. Resolved once at first use.
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
    arch = ""
    for k in d:
        if "." in k:
            arch = k.split(".", 1)[0]
            break
    for k, v in d.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            sys.stdout.write(f"{k}={v}\n")
            if arch and k.startswith(arch + "."):
                short = k[len(arch) + 1:]
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

_opt_offloaded_bytes() {
    local ngl="$1"
    local n_layer="$2"
    local total_model_bytes="$3"
    if [[ "$ngl" -le 0 ]]; then
        echo 0
    elif [[ "$ngl" -ge "$n_layer" ]]; then
        echo "$total_model_bytes"
    else
        awk -v ngl="$ngl" -v nl="$n_layer" -v total="$total_model_bytes" \
            'BEGIN { printf "%.0f", (ngl / nl) * total }'
    fi
}

_opt_total_memory() {
    local offloaded_bytes="$1"
    local kv_per_token="$2"
    local ctx_size="$3"
    local draft_bytes="$4"
    local draft_ctx="$5"
    local ssd_hot_mib="$6"
    local ssd_warm_mib="$7"
    local n_parallel="$8"
    local ubatch="$9"
    local moe_in_ram="${10:-0}"

    local kv_total=$(( ctx_size * kv_per_token * n_parallel ))
    local draft_kv_total=0
    if [[ $draft_ctx -gt 0 && $draft_bytes -gt 0 ]]; then
        draft_kv_total=$(( draft_ctx * kv_per_token / 4 * n_parallel ))
        local draft_kv_cap=$(( draft_bytes * 10 / 100 ))
        [[ $draft_kv_total -gt $draft_kv_cap ]] && draft_kv_total=$draft_kv_cap
    fi

    local ssd_bytes=$(( (ssd_hot_mib + ssd_warm_mib) * 1048576 ))
    # More conservative overhead: 2 GiB compute + 1 GiB system
    local compute_bytes=$(( 2 * 1024 * 1048576 + ubatch * 512 ))
    local system_bytes=$(( 1 * 1024 * 1048576 ))
    local raw=$(( offloaded_bytes + draft_bytes + kv_total + draft_kv_total + ssd_bytes + compute_bytes + system_bytes + moe_in_ram ))
    # Add 10% safety margin to avoid OOM
    echo $(( raw * 110 / 100 ))
}

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

_opt_start_optimistic() {
    # MOE strategy defaults – kept but ngl reduction will be secondary
    if [[ "${is_moe:-false}" == "true" && ${MODEL_BYTES:-0} -gt 0 ]]; then
        if [[ ${MODEL_BYTES} -gt ${GPU_BUDGET_BYTES:-0} ]]; then
            SOLVER_MOE_STRATEGY="cpu"
            SOLVER_LOAD_MODE="none"
        elif [[ ${MODEL_BYTES} -gt $(( ${GPU_BUDGET_BYTES:-0} * 95 / 100 )) ]]; then
            SOLVER_MOE_STRATEGY="residency"
            SOLVER_LOAD_MODE="mmap"
        fi
    fi

    local ctx_train=$(_opt_gguf context_length 32768)
    local ctx_cap=$(( ctx_train * 4 ))
    [[ $ctx_cap -gt 1048576 ]] && ctx_cap=1048576
    # Start at maximum possible context (ctx_cap), not training context.
    # This enables the optimistic-first approach: try max context, detune if needed.
    SOLVER_CTX_SIZE="$ctx_cap"
    if [[ $SOLVER_CTX_SIZE -lt $MIN_CTX ]]; then
        SOLVER_CTX_SIZE=$MIN_CTX
    fi

    SOLVER_K_TYPE="f16"
    SOLVER_V_TYPE="f16"

    local effective_tier="${LLAMA_HARDWARE_TIER:-standard}"
    [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]] && effective_tier="halo"
    SOLVER_TIER="$effective_tier"

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
        elif [[ "$(uname -s)" == "Darwin" ]]; then
            phys_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)
        else
            phys_cores=$(nproc)
            if [[ -f /sys/devices/system/cpu/smt/active ]] && [[ "$(cat /sys/devices/system/cpu/smt/active)" == "1" ]]; then
                phys_cores=$((phys_cores / 2))
            fi
        fi
    fi
    [[ -z "$phys_cores" || "$phys_cores" -lt 1 ]] && phys_cores=1

    case "$effective_tier" in
        halo)
            SOLVER_THREADS_BATCH="$phys_cores"
            SOLVER_THREADS="$(( phys_cores / 2 ))"
            ;;
        standard)
            SOLVER_THREADS_BATCH="$phys_cores"
            SOLVER_THREADS="$(( phys_cores / 2 ))"
            ;;
        handheld)
            SOLVER_THREADS_BATCH="$(( phys_cores / 2 ))"
            SOLVER_THREADS="$(( phys_cores / 4 ))"
            ;;
        *)
            SOLVER_THREADS_BATCH="$phys_cores"
            SOLVER_THREADS="$(( phys_cores / 2 ))"
            ;;
    esac
    [[ $SOLVER_THREADS_BATCH -lt 1 ]] && SOLVER_THREADS_BATCH=1
    [[ $SOLVER_THREADS -lt 1 ]] && SOLVER_THREADS=1

    SOLVER_SSD_ENABLE=true
    [[ "$effective_tier" == "halo" ]] && SOLVER_SSD_ENABLE=false
    case "$effective_tier" in
        halo)      SOLVER_SSD_HOT_RAM=2048; SOLVER_SSD_WARM_RAM=2048 ;;
        standard)  SOLVER_SSD_HOT_RAM=960;  SOLVER_SSD_WARM_RAM=1440 ;;
        handheld)  SOLVER_SSD_HOT_RAM=512;  SOLVER_SSD_WARM_RAM=768  ;;
        *)         SOLVER_SSD_HOT_RAM=960;  SOLVER_SSD_WARM_RAM=1440 ;;
    esac

    SOLVER_DRAFT_ENABLE=true
    SOLVER_DRAFT_N_MAX=8

    SOLVER_MOE_STRATEGY="${SOLVER_MOE_STRATEGY:-gpu}"
    SOLVER_MOE_RESIDENT_PER_LAYER=32
    SOLVER_MOE_PREWARM_TOP_K=16

    SOLVER_LOAD_MODE="${SOLVER_LOAD_MODE:-dio}"
    SOLVER_VK_NPS="${GGML_VK_NODES_PER_SUBMIT:-}"
    SOLVER_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-2048}"

    # Start with all layers offloaded
    SOLVER_NGL="$SOLVER_N_LAYER"
}

# New detune order: ctx first, then ngl, then others
_reduce_ctx() {
    local min_ctx=$MIN_CTX
    if [[ $SOLVER_CTX_SIZE -le $min_ctx ]]; then
        return 1
    fi
    # Step down: 262144 → 131072 → 65536 → 32768 → ...
    local ctx_values=(262144 131072 65536 32768 16384 8192)
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
    if [[ "$current" -le 0 ]]; then
        _SOLVER_DONE_reduce_ngl=1
        return 1
    fi
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
    [[ $SOLVER_SSD_HOT_RAM -le 256 && $SOLVER_SSD_WARM_RAM -le 256 ]] && return 1
    SOLVER_SSD_HOT_RAM=$(( SOLVER_SSD_HOT_RAM / 2 ))
    [[ $SOLVER_SSD_HOT_RAM -lt 256 ]] && SOLVER_SSD_HOT_RAM=256
    SOLVER_SSD_WARM_RAM=$(( SOLVER_SSD_WARM_RAM / 2 ))
    [[ $SOLVER_SSD_WARM_RAM -lt 256 ]] && SOLVER_SSD_WARM_RAM=256
    SOLVER_REASONS+=("SSD RAM: ${SOLVER_SSD_HOT_RAM}/${SOLVER_SSD_WARM_RAM} MiB")
    return 0
}

# Reduce both K and V cache to q8_0 together (paired quantization)
_reduce_kv_q8_0() {
    [[ "${_SOLVER_DONE_kv_q8_0:-0}" == "1" ]] && return 1
    SOLVER_K_TYPE="q8_0"
    SOLVER_V_TYPE="q8_0"
    _SOLVER_DONE_kv_q8_0=1
    SOLVER_REASONS+=("KV: q8_0/q8_0")
    return 0
}

# Reduce both K and V cache to q4_0 together (last resort before context reduction)
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

    _opt_read_gguf_meta "$model_path" || SOLVER_GGUF=()

    local n_layer=$(_opt_gguf block_count 32)
    SOLVER_N_LAYER="$n_layer"

    _opt_start_optimistic

    local draft_path=""
    [[ -n "${prof_dspark:-}" ]] && draft_path="$prof_dspark"
    [[ -z "$draft_path" && -n "${prof_dflash:-}" ]] && draft_path="$prof_dflash"
    local draft_bytes=0
    if [[ -n "$draft_path" && -f "$draft_path" ]]; then
        draft_bytes=$(stat -c%s "$draft_path" 2>/dev/null \
            || stat -f%z "$draft_path" 2>/dev/null || echo 0)
    fi

    local fai=$(_opt_gguf full_attention_interval 1)
    local n_attn
    n_attn=$(_opt_attn_layers "$n_layer" "$fai")

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

    # -------------------------------------------------------------------------
    # Phase 1: Find optimal (context, KV quality) pair using utility scoring
    # Utility = ctx * KV_weight, where KV_weight(f16)=5, q8=4, q4=1
    # Weights chosen to match user preference:
    # - 131072@q8 (utility=524288) > 262144@q4 (262144) and 65536@f16 (327680)
    # - At same context: f16(5) > q8(4) > q4(1)
    # - More context at same KV always wins
    # -------------------------------------------------------------------------
    local ctx_values=(262144 131072 65536 32768 16384 8192)
    # Weights chosen to match user preference:
# - 131072@q8 (utility=524288) > 262144@q4 (262144) and 65536@f16 (327680)
# - At same context: f16(5) > q8(4) > q4(1)
# - More context at same KV always wins
local kv_qualities=("f16/f16:5" "q8_0/q8_0:4" "q4_0/q4_0:1")
    local best_utility=-1
    local best_ctx=0
    local best_k_type=""
    local best_v_type=""
    local best_reason=""

    for ctx in "${ctx_values[@]}"; do
        [[ $ctx -lt $MIN_CTX ]] && continue
        [[ $ctx -gt $SOLVER_CTX_SIZE ]] && continue

        for kvq_weight in "${kv_qualities[@]}"; do
            local kvq="${kvq_weight%%:*}"
            local weight="${kvq_weight##*:}"
            local k_type="${kvq%%/*}"
            local v_type="${kvq##*/}"

            local kv_per_token_per_layer
            kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$k_type" "$v_type" "$hckv" "$kl" "$vl")
            local kv_per_token=$(( kv_per_token_per_layer * n_attn ))

            local offloaded_bytes
            offloaded_bytes=$(_opt_offloaded_bytes "$SOLVER_NGL" "$SOLVER_N_LAYER" "${MODEL_BYTES:-0}")

            local eff_draft_bytes=0
            [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"

            local mem_needed
            mem_needed=$(_opt_total_memory \
                "$offloaded_bytes" \
                "$kv_per_token" \
                "$ctx" \
                "$eff_draft_bytes" \
                "$ctx" \
                "$SOLVER_SSD_HOT_RAM" \
                "$SOLVER_SSD_WARM_RAM" \
                "${OVERRIDE_N_PARALLEL:-1}" \
                "$SOLVER_UBATCH" \
                0)

            if [[ $mem_needed -le ${GPU_BUDGET_BYTES:-0} ]] || [[ ${GPU_BUDGET_BYTES:-0} -le 0 ]]; then
                local utility=$(( ctx * weight ))
                # Tiebreaker: prefer higher KV weight (better quality) when utility equal
                if [[ $utility -gt $best_utility ]] || \
                   [[ $utility -eq $best_utility && $weight -gt ${best_weight:-0} ]]; then
                    best_utility=$utility
                    best_weight=$weight
                    best_ctx=$ctx
                    best_k_type="$k_type"
                    best_v_type="$v_type"
                    best_reason="ctx: ${ctx} KV: ${kvq} (utility=$utility, weight=$weight)"
                fi
            fi
        done
    done

    if [[ $best_utility -eq -1 ]]; then
        # Absolute fallback: MIN_CTX with q4_0/q4_0
        SOLVER_CTX_SIZE=$MIN_CTX
        SOLVER_K_TYPE="q4_0"
        SOLVER_V_TYPE="q4_0"
        SOLVER_REASONS+=("ctx: ${MIN_CTX} KV: q4_0/q4_0 (fallback)")
    else
        SOLVER_CTX_SIZE=$best_ctx
        SOLVER_K_TYPE=$best_k_type
        SOLVER_V_TYPE=$best_v_type
        SOLVER_REASONS+=("$best_reason")
    fi

    # Recompute kv_per_token for the chosen config
    local kv_per_token_per_layer
    kv_per_token_per_layer=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
    local kv_per_token=$(( kv_per_token_per_layer * n_attn ))

    # -------------------------------------------------------------------------
    # Phase 2: Apply remaining detunes if still over budget
    # Order: ngl -> ssd_ram -> draft -> ubatch
    # -------------------------------------------------------------------------
    local step_idx=0
    while [[ $step_idx -lt 50 ]]; do
        local offloaded_bytes
        offloaded_bytes=$(_opt_offloaded_bytes "$SOLVER_NGL" "$SOLVER_N_LAYER" "${MODEL_BYTES:-0}")

        local eff_draft_bytes=0
        [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"

        local mem_needed
        mem_needed=$(_opt_total_memory \
            "$offloaded_bytes" \
            "$kv_per_token" \
            "$SOLVER_CTX_SIZE" \
            "$eff_draft_bytes" \
            "$SOLVER_CTX_SIZE" \
            "$SOLVER_SSD_HOT_RAM" \
            "$SOLVER_SSD_WARM_RAM" \
            "${OVERRIDE_N_PARALLEL:-1}" \
            "$SOLVER_UBATCH" \
            0)

        if [[ $mem_needed -le ${GPU_BUDGET_BYTES:-0} ]] || [[ ${GPU_BUDGET_BYTES:-0} -le 0 ]]; then
            break
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

    local kv_cache_bytes=$(( SOLVER_CTX_SIZE * kv_per_token ))
    SOLVER_KV_CACHE_MIB=$(( kv_cache_bytes / 1048576 + 1024 ))

    SOLVER_PROFILE_NAME=$(_opt_pick_legacy_profile "$n_attn" "${MODEL_BYTES:-0}")
}

# Phase 2 detune steps (after context+KV are fixed)
_opt_detune_steps_phase2() {
    cat <<'STEPS'
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
    [[ -n "${OVERRIDE_CACHE_RAM:-}" ]] && { SOLVER_SSD_HOT_RAM="$OVERRIDE_CACHE_RAM"; SOLVER_SSD_WARM_RAM="$OVERRIDE_CACHE_RAM"; SOLVER_OVERRIDES+=("cache-ram"); }
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
