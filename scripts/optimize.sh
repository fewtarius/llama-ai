#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Optimistic-First Solver for llama.cpp Configuration
# =============================================================================
# Source this file (do not execute directly):
#   source scripts/optimize.sh
#
# Replaces the static preset table with a solver that:
#   1. Starts with the maximum-performance configuration (f16 KV, full SSD,
#      draft model, max ctx, max ubatch)
#   2. Computes the memory cost
#   3. Iteratively detunes by LEAST performance impact until the config fits
#      in the available memory budget
#
# Outputs are exposed via globals prefixed with SOLVER_*. llama-run.sh reads
# these after calling solve_optimal_config() and applies user overrides last.

[[ -n "${_LLAMA_OPTIMIZE_LOADED:-}" ]] && return 0
_LLAMA_OPTIMIZE_LOADED=1

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

    # Inline python helper that calls the reader, parses JSON, and emits
    # one "key=value" pair per scalar field. We avoid here-doc-in-$(...) nesting
    # by writing the helper to a temp file first.
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

_opt_total_memory() {
    local model_bytes=$1
    local kv_per_token=$2
    local ctx_size=$3
    local draft_bytes=${4:-0}
    local draft_ctx=${5:-0}
    local ssd_hot_mib=${6:-0}
    local ssd_warm_mib=${7:-0}
    local n_parallel=${8:-1}
    local ubatch=${9:-2048}
    local moe_in_ram=${10:-0}

    local kv_total=$(( (ctx_size * kv_per_token + draft_ctx * kv_per_token / 4) * n_parallel ))
    local ssd_bytes=$(( (ssd_hot_mib + ssd_warm_mib) * 1048576 ))
    local compute_mib=$(( 4 * 1024 + ubatch * 2 ))
    [[ $compute_mib -gt $(( 12 * 1024 )) ]] && compute_mib=$(( 12 * 1024 ))
    local compute_bytes=$(( compute_mib * 1048576 ))
    local system_bytes=$(( 2 * 1024 * 1048576 ))
    echo $(( model_bytes + draft_bytes + kv_total + ssd_bytes + compute_bytes + system_bytes + moe_in_ram ))
}

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

_opt_start_optimistic() {
    # MOE strategy defaults: match the legacy `_apply_moe_streaming` logic.
    # If the model exceeds the GPU budget, pin everything to host RAM via
    # --cpu-moe + --load-mode none (no mmap, no GTT pages). If it's just
    # under 95% of the budget, use --moe-expert-residency (mmap + madvise
    # tracking) which avoids page-fault pathology on Flip's 6 GB VRAM
    # carveout without paying the full load-time cost of --load-mode none.
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
    SOLVER_CTX_SIZE="$ctx_train"
    local ctx_cap=$(( ctx_train * 4 ))
    [[ $ctx_cap -gt 1048576 ]] && ctx_cap=1048576
    [[ $SOLVER_CTX_SIZE -gt $ctx_cap ]] && SOLVER_CTX_SIZE=$ctx_cap

    SOLVER_K_TYPE="f16"
    SOLVER_V_TYPE="f16"

    # Determine effective tier. LLAMA_IS_STRIX_HALO is the strongest signal
    # (PCI ID is exclusive); it overrides the VRAM-based LLAMA_HARDWARE_TIER
    # classification. This is important on Nimo: BIOS gives 512 MiB VRAM
    # which would otherwise mis-classify as standard.
    local effective_tier="${LLAMA_HARDWARE_TIER:-standard}"
    [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]] && effective_tier="halo"
    SOLVER_TIER="$effective_tier"

    case "$effective_tier" in
        # ubatch=2048: HTTP-server benchmark on Nimo (Qwen3.6-35B-A3B
        # Q8_K_XL, ctx=262144) gives 67.8 t/s decode vs 18.3 t/s at
        # ubatch=512. MTP draft expansion benefits from the larger
        # ubatch. batch=2048 is the legacy default across the Strix Halo
        # and 780M hardware range; we don't assume Nimo-specific headroom.
        # Standard/handheld use smaller values to fit smaller APUs
        # (e.g. Steam Deck 780M, 16 GB VRAM+GTT) where ubatch=2048 OOMs.
        halo)      SOLVER_UBATCH=2048; SOLVER_BATCH=2048 ;;
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
    SOLVER_THREADS_BATCH="$phys_cores"
    local gen_threads=$(( phys_cores - 2 ))
    [[ $gen_threads -lt 1 ]] && gen_threads=1
    SOLVER_THREADS="$gen_threads"

    SOLVER_SSD_ENABLE=true
    # SSD cache: on halo tier, disable. Benchmarks (Qwen3.6-35B-A3B and
    # Qwen3.5-122B-A10B on Strix Halo) show SSD's constant KV serialization
    # overhead costs 20-30% of prompt throughput with no measurable win
    # for typical workloads. The legacy preset table makes the same call
    # for all halo profiles. On non-halo tiers the SSD offload helps
    # because the GPU budget is tight enough that prompt cache reuse is
    # worth the serialization cost.
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

    SOLVER_LOAD_MODE="${SOLVER_LOAD_MODE:-mmap}"
    SOLVER_VK_NPS="${GGML_VK_NODES_PER_SUBMIT:-}"
    SOLVER_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-2048}"
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

_reduce_ctx() {
    local ctx_values=(262144 131072 65536 32768 16384 8192)
    local c
    for c in "${ctx_values[@]}"; do
        if [[ $SOLVER_CTX_SIZE -gt $c ]]; then
            SOLVER_CTX_SIZE=$c
            SOLVER_REASONS+=("ctx: ${c}")
            return 0
        fi
    done
    return 1
}

_reduce_ubatch() {
    [[ $SOLVER_UBATCH -le 512 ]] && return 1
    SOLVER_UBATCH=$(( SOLVER_UBATCH / 2 ))
    [[ $SOLVER_BATCH -gt $SOLVER_UBATCH ]] && SOLVER_BATCH=$SOLVER_UBATCH
    SOLVER_REASONS+=("ubatch: ${SOLVER_UBATCH}")
    return 0
}

_reduce_threads() {
    [[ $SOLVER_THREADS_BATCH -le 4 ]] && return 1
    SOLVER_THREADS_BATCH=$(( SOLVER_THREADS_BATCH - 2 ))
    [[ $SOLVER_THREADS_BATCH -lt 4 ]] && SOLVER_THREADS_BATCH=4
    local gen=$(( SOLVER_THREADS_BATCH - 2 ))
    [[ $gen -lt 1 ]] && gen=1
    SOLVER_THREADS=$gen
    SOLVER_REASONS+=("threads: ${SOLVER_THREADS_BATCH}/${SOLVER_THREADS}")
    return 0
}

_v_q8_0() {
    [[ "${_SOLVER_DONE_v_q8_0:-0}" == "1" ]] && return 1
    SOLVER_V_TYPE="q8_0"
    _SOLVER_DONE_v_q8_0=1
    SOLVER_REASONS+=("V: q8_0")
    return 0
}

_v_q4_0() {
    [[ "${_SOLVER_DONE_v_q4_0:-0}" == "1" ]] && return 1
    SOLVER_V_TYPE="q4_0"
    _SOLVER_DONE_v_q4_0=1
    SOLVER_REASONS+=("V: q4_0")
    return 0
}

_k_q8_0() {
    [[ "${_SOLVER_DONE_k_q8_0:-0}" == "1" ]] && return 1
    SOLVER_K_TYPE="q8_0"
    _SOLVER_DONE_k_q8_0=1
    SOLVER_REASONS+=("K: q8_0")
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

_moe_residency() {
    [[ "${_SOLVER_DONE_moe_residency:-0}" == "1" || "$SOLVER_MOE_STRATEGY" != "gpu" ]] && return 1
    [[ "${is_moe:-false}" != "true" ]] && return 1
    SOLVER_MOE_STRATEGY="residency"
    SOLVER_LOAD_MODE="mmap"
    _SOLVER_DONE_moe_residency=1
    SOLVER_REASONS+=("MoE: residency")
    return 0
}

_moe_cpu() {
    [[ "${_SOLVER_DONE_moe_cpu:-0}" == "1" || "$SOLVER_MOE_STRATEGY" == "cpu" ]] && return 1
    [[ "${is_moe:-false}" != "true" ]] && return 1
    SOLVER_MOE_STRATEGY="cpu"
    SOLVER_LOAD_MODE="none"
    _SOLVER_DONE_moe_cpu=1
    SOLVER_REASONS+=("MoE: cpu-moe + load-mode none")
    return 0
}

# For dense models that don't fit: enable --fit so llama.cpp can split
# layers between GPU and CPU. This is the last-resort detune for models
# larger than the GPU budget.
_dense_fit() {
    [[ "${_SOLVER_DONE_dense_fit:-0}" == "1" ]] && return 1
    [[ "${is_moe:-false}" == "true" ]] && return 1
    SOLVER_NGL=-1
    _SOLVER_DONE_dense_fit=1
    SOLVER_REASONS+=("dense-fit: ngl=auto")
    return 0
}

_opt_detune_steps() {
    cat <<'STEPS'
_reduce_ssd_ram
_reduce_ctx
_reduce_ubatch
_reduce_threads
_v_q8_0
_v_q4_0
_k_q8_0
_drop_draft
_drop_ssd
_dense_fit
STEPS
}

solve_optimal_config() {
    local model_path="$1"

    # Clear the once-per-config detune flags so re-running the solver
    # (e.g. after a manual override change) starts fresh.
    _SOLVER_DONE_v_q8_0=0
    _SOLVER_DONE_v_q4_0=0
    _SOLVER_DONE_k_q8_0=0
    _SOLVER_DONE_drop_draft=0
    _SOLVER_DONE_drop_ssd=0
    _SOLVER_DONE_moe_residency=0
    _SOLVER_DONE_moe_cpu=0
    _SOLVER_DONE_dense_fit=0

    _opt_read_gguf_meta "$model_path" || SOLVER_GGUF=()
    _opt_start_optimistic

    local draft_path=""
    [[ -n "${prof_dspark:-}" ]] && draft_path="$prof_dspark"
    [[ -z "$draft_path" && -n "${prof_dflash:-}" ]] && draft_path="$prof_dflash"
    local draft_bytes=0
    if [[ -n "$draft_path" && -f "$draft_path" ]]; then
        draft_bytes=$(stat -c%s "$draft_path" 2>/dev/null \
            || stat -f%z "$draft_path" 2>/dev/null || echo 0)
    fi

    local n_layer=$(_opt_gguf block_count 32)
    local fai=$(_opt_gguf full_attention_interval 1)
    local n_attn
    n_attn=$(_opt_attn_layers "$n_layer" "$fai")
    local hckv=$(_opt_gguf head_count_kv 2)
    local kl=$(_opt_gguf key_length 256)
    local vl=$(_opt_gguf value_length 256)
    local kv_per_token
    kv_per_token=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
    kv_per_token=$(( kv_per_token * n_attn ))

    SOLVER_REASONS=()
    SOLVER_DRAFT_PATH="$draft_path"
    local step_idx=0
    while [[ $step_idx -lt 50 ]]; do
        local eff_draft_bytes=0
        [[ "$SOLVER_DRAFT_ENABLE" == "true" ]] && eff_draft_bytes="$draft_bytes"

        local mem_needed
        mem_needed=$(_opt_total_memory \
            "${MODEL_BYTES:-0}" \
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
                if [[ "$step_fn" == "_v_q8_0" || "$step_fn" == "_v_q4_0" || "$step_fn" == "_k_q8_0" ]]; then
                    kv_per_token=$(_opt_layer_kv_bytes_per_token "$SOLVER_K_TYPE" "$SOLVER_V_TYPE" "$hckv" "$kl" "$vl")
                    kv_per_token=$(( kv_per_token * n_attn ))
                fi
                break
            fi
        done < <(_opt_detune_steps)

        [[ $applied -eq 0 ]] && break
        step_idx=$(( step_idx + 1 ))
    done

    SOLVER_PROFILE_NAME=$(_opt_pick_legacy_profile "$n_attn" "${MODEL_BYTES:-0}")
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
    # User-set variables (set via CLI flags). Use the actual user-set value,
    # NOT the LLAMA_* default that might still hold the default value.
    [[ -n "${USER_CTX_SIZE:-}" ]] && { SOLVER_CTX_SIZE="$CTX_SIZE"; SOLVER_OVERRIDES+=("ctx-size"); }
    [[ -n "${USER_KV_CACHE_TYPE:-}" ]] && { SOLVER_K_TYPE="$KV_CACHE_TYPE_K"; SOLVER_V_TYPE="$KV_CACHE_TYPE_V"; SOLVER_OVERRIDES+=("kv-cache-type"); }
    [[ -n "${KV_CACHE_K_OVERRIDE:-}" ]] && { SOLVER_K_TYPE="$KV_CACHE_K_OVERRIDE"; SOLVER_OVERRIDES+=("cache-type-k"); }
    [[ -n "${KV_CACHE_V_OVERRIDE:-}" ]] && { SOLVER_V_TYPE="$KV_CACHE_V_OVERRIDE"; SOLVER_OVERRIDES+=("cache-type-v"); }
    [[ -n "${LLAMA_THREADS_OVERRIDE:-}" ]] && { SOLVER_THREADS="$LLAMA_THREADS_OVERRIDE"; SOLVER_THREADS_BATCH="$LLAMA_THREADS_OVERRIDE"; SOLVER_OVERRIDES+=("threads"); }
    # If no explicit user override, use LLAMA_THREADS (auto-detected by
    # detect-gpu.sh). This includes SMT threads (e.g., 32 on Nimo's 16-core
    # Strix Halo). The legacy preset table uses LLAMA_THREADS directly, and
    # benchmark data on Qwen3.5-122B shows 32 threads gives +40% prefill
    # vs 14 because prefill is compute-bound and SMT helps saturate cores.
    if [[ -z "${LLAMA_THREADS_OVERRIDE:-}" && -n "${LLAMA_THREADS:-}" ]]; then
        SOLVER_THREADS="$LLAMA_THREADS"
        SOLVER_THREADS_BATCH="$LLAMA_THREADS"
        # Don't add to SOLVER_OVERRIDES (it's the default, not a user override).
    fi
    [[ -n "${MOE_UBATCH_OVERRIDE:-}" ]] && { SOLVER_UBATCH="$MOE_UBATCH_OVERRIDE"; SOLVER_OVERRIDES+=("ubatch"); }
    [[ -n "${OVERRIDE_UBATCH_SIZE:-}" ]] && { SOLVER_UBATCH="$OVERRIDE_UBATCH_SIZE"; SOLVER_OVERRIDES+=("ubatch-size"); }
    [[ -n "${OVERRIDE_CACHE_RAM:-}" ]] && { SOLVER_SSD_HOT_RAM="$OVERRIDE_CACHE_RAM"; SOLVER_SSD_WARM_RAM="$OVERRIDE_CACHE_RAM"; SOLVER_OVERRIDES+=("cache-ram"); }
    # SSD hot/warm RAM defaults (LLAMA_SSD_HOT_RAM / LLAMA_SSD_WARM_RAM) are
    # not treated as user overrides -- they're defaults the legacy code used
    # for SSD buffer sizes. The solver decides the right value based on tier.
    # To force a specific value, use --cache-ssd-hot-ram / --cache-ssd-warm-ram
    # CLI flags (which set SSD_HOT_RAM / SSD_WARM_RAM in the integration code).
    if [[ "${_SSD_DISABLE:-false}" == "true" ]]; then
        SOLVER_SSD_ENABLE=false
        SOLVER_OVERRIDES+=("no-ssd-cache")
    fi
    [[ "${OVERRIDE_FIT:-}" == "on" ]] && { SOLVER_NGL=-1; SOLVER_OVERRIDES+=("--fit on"); }
    [[ -n "${OVERRIDE_REASONING_BUDGET:-}" ]] && { SOLVER_REASONING_BUDGET="$OVERRIDE_REASONING_BUDGET"; SOLVER_OVERRIDES+=("reasoning-budget"); }
}
