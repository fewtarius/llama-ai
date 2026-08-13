#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Llama.cpp Unified Runner — Fully adaptive (no hardcoded sizes)
# =============================================================================
# Auto-scans ./models for available GGUF files
# Supports Vulkan, ROCm/HIP, Metal, CPU backends
# All performance‑critical values are detected at runtime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Auto-detect GPU and system resources
source "$PROJECT_ROOT/scripts/detect-gpu.sh"

# =============================================================================
# Centralised defaults (all overridable via environment)
# =============================================================================
: "${ENABLE_CONTEXT_SHIFT:=1}"
: "${KV_CACHE_K_OVERRIDE:=}"
: "${KV_CACHE_V_OVERRIDE:=}"
: "${MOE_UBATCH_OVERRIDE:=}"
: "${CACHE_RAM_OVERRIDE:=}"
: "${LLAMA_THREADS:=}"                       # auto-detect if empty
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
: "${LLAMA_TOP_P:=0.85}"
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

# Speculative decoding tuning. Each spec type has a tuned default for the
# empirically-best `n_max`. Set LLAMA_SPEC_DRAFT_N_MAX (a single value) as
# a fallback that applies to every spec type when its per-type variant is
# left at the hardcoded default. Per-type values win over the global
# fallback (set LLAMA_SPEC_DRAFT_N_MAX_DSPARK=5 to override only DSpark).
: "${LLAMA_SPEC_DRAFT_N_MAX:=}"
: "${LLAMA_SPEC_DRAFT_N_MAX_DSPARK:=${LLAMA_SPEC_DRAFT_N_MAX:-3}}"  # DeepSeek Spark - tuned 2026-08
: "${LLAMA_SPEC_DRAFT_N_MAX_MTP:=${LLAMA_SPEC_DRAFT_N_MAX:-2}}"     # Qwen 3.5/3.6 MTP head - Unsloth rec
: "${LLAMA_SPEC_DRAFT_N_MAX_DFLASH:=${LLAMA_SPEC_DRAFT_N_MAX:-7}}"  # Laguna block-diffusion - tuned 2026-08

# Platform detection
if [[ "$(uname -s)" == "Darwin" ]]; then
    IS_DARWIN=true
    [[ "$(uname -m)" == "arm64" ]] && IS_DARWIN_ARM=true || IS_DARWIN_ARM=false
else
    IS_DARWIN=false; IS_DARWIN_ARM=false
fi

MODEL_DIR="$PROJECT_ROOT/models"

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
    [[ "$(uname -s)" == "Darwin" ]] && BACKEND="metal" || BACKEND="vulkan"
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

# Structured RAM-budget breakdown. Populated by compute_ram_budget_breakdown()
# so compute_cache_ram() and print_profile_summary() can both report the
# per-component accounting. Format: one "label=bytes" entry per line.
# Caller sums the bytes; the labels are for display.
_RAM_BUDGET_BREAKDOWN=""

# Sum the bytes values in a "label=bytes" breakdown (one entry per line).
# Echoes the integer total. Lines that don't parse cleanly are skipped
# (defensive against upstream whitespace / comment edits).
_sum_ram_breakdown() {
    local total=0
    while IFS='=' read -r label bytes; do
        [[ -z "$label" || "$label" == \#* ]] && continue
        [[ "$bytes" =~ ^[0-9]+$ ]] || continue
        total=$(( total + bytes ))
    done <<< "$1"
    echo "$total"
}

# Compute the structured RAM-budget breakdown (one "label=bytes" entry per
# line, stored in $_RAM_BUDGET_BREAKDOWN). Every component represents RAM
# the server will consume on top of the target model's weights; cache-ram
# must be sized so the sum of all entries plus MODEL_BYTES still fits
# within GPU_BUDGET_BYTES.
#
# Components:
#   ssd_hot       SSD KV cache hot-tier RAM buffer (LLAMA_SSD_HOT_RAM MiB)
#   ssd_warm      SSD KV cache warm-tier RAM buffer (LLAMA_SSD_WARM_RAM MiB)
#   draft_model   DSpark / DFlash draft model file size
#   draft_kv      Draft model KV cache (target ctx × draft K/V types)
#   mtp_context   MTP head speculative decoding context overhead
#   ngram_cache   n-gram speculative decoding tables (ngram-simple/mod/map/cache)
#   lookup_cache  --lookup-cache-static/-dynamic in-memory corpus
#   parallel_kv   Extra KV cache for --np > 1 slots
#   cpu_moe_model Target model bytes (only when --cpu-moe + --load-mode none is predicted)
#   general       General reserve (scales 4-11 GiB with --ubatch-size)
#
# Always-on (every profile): `general`.
# Conditional on profile state: every other component.
compute_ram_budget_breakdown() {
    local budget="${GPU_BUDGET_BYTES:-0}"
    local total_mem=$(get_total_memory_bytes)
    local breakdown=""

    # SSD hot/warm RAM buffers (only when SSD cache is enabled by profile).
    # Default hot=960 MiB + warm=1440 MiB = 2400 MiB always-on RAM when SSD
    # is active. halo-* profiles disable SSD entirely; std-* profiles enable
    # it by default. See _apply_ssd_defaults() for the value plumbing.
    if [[ "${_SSD_DISABLE:-false}" != "true" ]]; then
        local ssd_hot_mib="${LLAMA_SSD_HOT_RAM:-0}"
        local ssd_warm_mib="${LLAMA_SSD_WARM_RAM:-0}"
        breakdown+="ssd_hot=$(( ssd_hot_mib * 1048576 ))"$'\n'
        breakdown+="ssd_warm=$(( ssd_warm_mib * 1048576 ))"$'\n'
    fi

    # Draft model file size + KV cache (DSpark / DFlash). DSpark uses a
    # small draft (1-3 GiB typical, e.g. DeepSeek-V4-Flash DSpark Q8_0);
    # DFlash uses the full decoder-side architecture (5-15 GiB typical,
    # e.g. Laguna-S-2.1 DFlash BF16). Both get their own KV cache sized
    # to the TARGET's n_ctx at --cache-type-k-draft/-v-draft (default
    # f16, see common.h params.speculative.draft.cache_type_k default).
    # Confirmed in tools/server/server-context.cpp:1209-1259 (ctx_dft
    # uses cparams_dft = common_context_params_to_llama(params_dft)
    # which sets cparams.n_ctx = params.n_ctx = target n_ctx).
    #
    # Estimate draft KV cache as 30% of draft file size: covers typical
    # ctx*layers*embd*K+V bytes for small quant drafts at default ctx.
    # Min 256 MiB so tiny drafts still get sensible headroom.
    #
    # draft path lookup: prefer the pre-detected globals (prof_dspark /
    # prof_dflash, populated in assign_profile()), but fall back to
    # parsing EXTRA_SERVER_ARGS when this function is called from
    # print_profile_summary() -- at which point the locals from
    # assign_profile() have already gone out of scope.
    local draft_path=""
    [[ -n "${prof_dspark:-}" ]] && draft_path="$prof_dspark"
    [[ -z "$draft_path" && -n "${prof_dflash:-}" ]] && draft_path="$prof_dflash"
    if [[ -z "$draft_path" && -n "${EXTRA_SERVER_ARGS:-}" ]]; then
        # Fallback: extract from " -md <path>" already in EXTRA_SERVER_ARGS.
        # Matches both DSpark and DFlash drafts (both add "-md <draft>").
        draft_path=$(echo "${EXTRA_SERVER_ARGS:-}" | sed -nE 's/.* -md ([^ ]+).*/\1/p' | head -1)
        [[ "$draft_path" == *"-spec-type"* ]] && draft_path=""  # safety: -md might match other args
    fi
    if [[ -n "$draft_path" && -f "$draft_path" ]]; then
        local draft_bytes=$(stat -c%s "$draft_path" 2>/dev/null || stat -f%z "$draft_path" 2>/dev/null || echo 0)
        breakdown+="draft_model=$draft_bytes"$'\n'
        local draft_kv=$(( draft_bytes * 30 / 100 ))
        [[ $draft_kv -lt $(( 256 * 1048576 )) ]] && draft_kv=$(( 256 * 1048576 ))
        [[ $draft_kv -gt $(( 8 * 1048576 * 1024 / 10 )) ]] && draft_kv=$(( 8 * 1048576 * 1024 / 10 ))  # cap at 800 MiB
        breakdown+="draft_kv=$draft_kv"$'\n'
    fi

    # MTP context overhead. MTP creates an LLAMA_CONTEXT_TYPE_MTP context
    # on the target model (server-context.cpp:1264-1282) with its own KV
    # cache at draft K/V types (default f16). The context size = target
    # n_ctx (no separate spec-draft-ctx-size flag exists). Estimate as
    # 10% of target MODEL_BYTES with [1 GiB, 4 GiB] bounds -- covers KV
    # cache + compute buffers + scratch for typical MTP head counts
    # (nextn_predict_layers=1-4).
    if [[ "${is_mtp:-false}" == "true" ]]; then
        local mtp_ctx=$(( MODEL_BYTES / 10 ))
        [[ $mtp_ctx -lt $(( 1 * 1048576 * 1024 )) ]] && mtp_ctx=$(( 1 * 1048576 * 1024 ))
        [[ $mtp_ctx -gt $(( 4 * 1048576 * 1024 )) ]] && mtp_ctx=$(( 4 * 1048576 * 1024 ))
        breakdown+="mtp_context=$mtp_ctx"$'\n'
    fi

    # n-gram speculative decoding tables (ngram-simple, ngram-mod,
    # ngram-map, ngram-cache). ngram_simple/mod/map store hash tables
    # in RAM; --spec-ngram-mod-n-max (default 64) and
    # --spec-ngram-simple-size-m (default 48) bound growth (common.h
    # common_params_speculative_ngram_mod / _ngram_map). Small flat
    # 256 MiB allocation.
    if [[ "${EXTRA_SERVER_ARGS:-}" == *"ngram"* ]] || \
       [[ "${LLAMA_SPEC_TYPE:-}" == *"ngram"* ]]; then
        breakdown+="ngram_cache=$(( 256 * 1048576 ))"$'\n'
    fi

    # Lookup cache (--lookup-cache-static/-dynamic). In-memory text
    # corpus hash table, scales with corpus size. Small flat 256 MiB
    # allocation -- only on if paths are explicitly set.
    if [[ -n "${LLAMA_LOOKUP_CACHE_STATIC:-}" ]] || [[ -n "${LLAMA_LOOKUP_CACHE_DYNAMIC:-}" ]]; then
        breakdown+="lookup_cache=$(( 256 * 1048576 ))"$'\n'
    fi

    # Additional parallel slots (--np > 1). Each extra slot needs its
    # own KV cache sized to CTX_SIZE × n_layer × n_embd × K+V bytes.
    # Without reading GGUF metadata we estimate 5% of MODEL_BYTES per
    # slot (dense 7B at 32k ctx ≈ 0.5 GiB; dense 70B at 32k ctx ≈
    # 5 GiB; the 5%-of-model heuristic matches the middle of that
    # range). Floor 512 MiB per slot so small models don't under-budget.
    local n_parallel="${OVERRIDE_N_PARALLEL:-1}"
    [[ -z "$n_parallel" || $n_parallel -lt 1 ]] && n_parallel=1
    if [[ $n_parallel -gt 1 ]]; then
        local extra_slots=$(( n_parallel - 1 ))
        local per_slot=$(( MODEL_BYTES * 5 / 100 ))
        [[ $per_slot -lt $(( 512 * 1048576 )) ]] && per_slot=$(( 512 * 1048576 ))
        breakdown+="parallel_kv=$(( extra_slots * per_slot ))"$'\n'
    fi

    # General reserve: compute buffers, llama.cpp internal state, sampler
    # chains, grammar buffers, scratch for matmul workspaces, slot state
    # per parallel slot. Scales with --ubatch-size because compute buffer
    # pools grow with batch size (flash-attn working memory roughly doubles
    # per ubatch step). Floor 4 GiB for std-* profiles, ~11 GiB for
    # halo-moe-large (ubatch=4096, batch=4096 -- flash-attn forward
    # buffers hit 1-2 GiB alone, compute pool another 4-6 GiB).
    #   ubatch=512  -> 4 GiB  (ssm profile)
    #   ubatch=1024 -> 5 GiB  (std-moe-large / std-dense-*)
    #   ubatch=2048 -> 7 GiB  (large dense)
    #   ubatch=4096 -> 11 GiB (halo-moe-large)
    # 4 GiB floor keeps std-moe-large on Flip (26 GiB OS-visible, 20 GiB
    # MoE + 2.4 GiB SSD + 5 GiB reserve) at 0 MiB cache-ram when full
    # --cpu-moe + --load-mode none is in play -- matches the LTM-stated
    # 2 GiB cap with system overhead trimmed. Was 3 GiB flat previously.
    local ubatch=1024
    if [[ -n "${OVERRIDE_BATCH_SIZE:-}" ]]; then
        local _ub
        _ub=$(echo "${OVERRIDE_BATCH_SIZE:-}" | sed -nE 's/.*--ubatch-size ([0-9]+).*/\1/p' | head -1)
        [[ -n "$_ub" && "$_ub" =~ ^[0-9]+$ ]] && ubatch="$_ub"
    fi
    local general_reserve_mib=$(( 4 * 1024 + (ubatch * 2) ))
    [[ $general_reserve_mib -gt $(( 12 * 1024 )) ]] && general_reserve_mib=$(( 12 * 1024 ))
    breakdown+="general=$(( general_reserve_mib * 1048576 ))"$'\n'

    # MoE --cpu-moe + --load-mode none prediction. If the model will be
    # loaded fully into RAM (MoE + exceeds GPU budget + fits in system
    # RAM), subtract its size from the cache-ram budget. This avoids
    # proposing a cache-ram that overlaps with the in-RAM model and
    # crashes the server at startup. Without this, the std-moe-large
    # profile on Flip (24 GiB GPU budget, 20+ GiB Q4_K_XL MoE) would
    # propose a cache-ram that's already accounted for as the model's
    # own RAM footprint.
    if [[ "${is_moe:-false}" == "true" && $MODEL_BYTES -gt $budget && $MODEL_BYTES -lt $total_mem ]]; then
        breakdown+="cpu_moe_model=$MODEL_BYTES"$'\n'
    fi

    # Strip trailing newline so callers can store cleanly.
    _RAM_BUDGET_BREAKDOWN="${breakdown%$'\n'}"
}

# Compute --cache-ram in MiB, leaving room for every enabled RAM consumer
# tracked by compute_ram_budget_breakdown() plus the model + a 6 GiB
# general reserve. Caches the breakdown in $_RAM_BUDGET_BREAKDOWN for the
# profile summary to display.
#
# Base formula (preserved from previous version):
#     max_cache = GPU_BUDGET_BYTES - MODEL_BYTES - reserve
# where reserve is now the SUM of every component in the breakdown (not a
# flat 3 GiB). For typical std-* profiles this nets slightly less cache-ram
# (e.g. 26-3-20 = 3 → 26-20-2.4-6 = -2.4 = 0 MiB on std-moe-large Flip),
# which matches the actual --cpu-moe + --load-mode none RAM footprint
# (no room for in-memory checkpoints when model occupies everything).
compute_cache_ram() {
    compute_ram_budget_breakdown
    local breakdown_total=$(_sum_ram_breakdown "$_RAM_BUDGET_BREAKDOWN")
    local model_bytes="${MODEL_BYTES:-0}"
    local budget="${GPU_BUDGET_BYTES:-0}"
    local cache_mib=8192
    if [[ $budget -gt 0 ]]; then
        # If the breakdown already accounts for MODEL_BYTES (cpu-moe
        # case -- model loaded into RAM via --load-mode none), skip the
        # extra `- model_bytes` to avoid double-counting. Otherwise the
        # model is on GPU and we subtract it from the GPU budget as
        # before.
        local has_cpu_moe="false" _lbl
        while IFS='=' read -r _lbl _; do
            [[ "$_lbl" == "cpu_moe_model" ]] && has_cpu_moe="true"
        done <<< "$_RAM_BUDGET_BREAKDOWN"
        local max_cache
        if [[ "$has_cpu_moe" == "true" ]]; then
            max_cache=$((budget - breakdown_total))
        else
            max_cache=$((budget - breakdown_total - model_bytes))
        fi
        if [[ $max_cache -gt 0 ]]; then
            cache_mib=$((max_cache / 1048576))
        else
            cache_mib=0
        fi
    else
        log_warn "GPU budget not detected; using conservative cache-ram: ${cache_mib} MiB"
    fi
    echo "$cache_mib"
}

compute_kv_types() {
    local k="${LLAMA_KV_CACHE_TYPE_K}" v="${LLAMA_KV_CACHE_TYPE_V}"
    [[ -n "$KV_CACHE_K_OVERRIDE" ]] && k="$KV_CACHE_K_OVERRIDE"
    [[ -n "$KV_CACHE_V_OVERRIDE" ]] && v="$KV_CACHE_V_OVERRIDE"
    echo "$k $v"
}

compute_ubatch() {
    local model_gb=$((MODEL_BYTES / 1073741824))
    local ub=512
    [[ "${is_moe:-false}" == "true" || $model_gb -gt 60 ]] && ub=1024
    [[ -n "$MOE_UBATCH_OVERRIDE" ]] && ub="$MOE_UBATCH_OVERRIDE"
    echo "$ub"
}

check_ac_power() {
    local on_ac=0
    if [[ -d /sys/class/power_supply ]]; then
        for f in /sys/class/power_supply/{AC,Mains,ADP}*/online; do
            [[ -r "$f" && "$(cat "$f" 2>/dev/null)" == "1" ]] && on_ac=1
        done
    fi
    if command -v pmset &>/dev/null && [[ $on_ac -eq 0 ]]; then
        pmset -g batt 2>/dev/null | grep -q "AC Power" && on_ac=1
    fi
    return $((1 - on_ac))
}

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$1"; }
log_ok()    { printf '%b[OK]%b   %s\n' "$GREEN" "$NC" "$1"; }
log_warn()  { printf '%b[WARN]%b  %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1"; }

# -----------------------------------------------------------------------------
# Profile engine – replaces all individual _preset_* functions
# -----------------------------------------------------------------------------
profile_name=""

assign_profile() {
    local model_path="$1"
    local filename=$(basename "$model_path")
    local size_bytes

    # Calculate total model size (handles sharded GGUF)
    if [[ "$model_path" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        local shard_base="${model_path%-${BASH_REMATCH[1]}-of-${BASH_REMATCH[2]}.gguf}"
        local shard_count=$((10#${BASH_REMATCH[2]}))
        size_bytes=0
        for ((i=1; i<=shard_count; i++)); do
            local sf=$(printf "%s-%05d-of-%05d.gguf" "$shard_base" "$i" "$shard_count")
            if [[ -f "$sf" ]]; then
                size_bytes=$((size_bytes + $(stat -c%s "$sf" 2>/dev/null || stat -f%z "$sf" 2>/dev/null || echo 0)))
            fi
        done
        [[ $size_bytes -eq 0 ]] && size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null || echo 0)
    else
        size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null || echo 0)
    fi
    MODEL_BYTES="$size_bytes"
    local size_gb=$((size_bytes / 1073741824))

    local is_strix_halo="false"
    [[ "${LLAMA_IS_STRIX_HALO:-0}" == "1" ]] && is_strix_halo="true"

    # Initialize MoE/SSM flags (globals so _scan_gguf_arch can toggle them).
    # Filename pattern + GGUF header probe may flip them below; without an
    # explicit default, dense models (no experts, no SSM) leave is_moe/is_ssm
    # unset and `set -u` blows up at line 397 (unbound variable).
    is_moe="false"
    is_ssm="false"

    # Architecture detection
    if echo "$filename" | grep -qiE "moe|a3b|a8b|flash|expert|gpt-oss"; then
        is_moe=true
    fi
    if echo "$filename" | grep -qiE "ssm|mamba|jamba|falcon-h1|rwkv"; then
        is_ssm=true
    fi
    # Initialise is_mtp explicitly so unset state under `set -u` is safe;
    # _scan_gguf_arch() flips it to true when the GGUF header carries
    # `nextn_predict_layers` plus a `nextn.eh_proj` tensor (Qwen 3.5/3.6
    # via qwen35moe, DeepSeek-V4, etc.). See _scan_gguf_arch() for details.
    is_mtp="false"
    _scan_gguf_arch "$model_path"   # sets is_moe / is_ssm / is_mtp

    # Threads
    if [[ -z "${LLAMA_THREADS:-}" ]]; then
        local phys=$(_get_physical_cores)
        THREADS=$((phys - 2))
        [[ $THREADS -lt 1 ]] && THREADS=1
        log_info "Physical cores: $phys, using $THREADS compute threads"
    else
        THREADS="$LLAMA_THREADS"
        log_info "Using user‑set threads: $THREADS"
    fi

    # KV cache types
    read -r KV_CACHE_TYPE_K KV_CACHE_TYPE_V <<< "$(compute_kv_types)"

    # Ubatch / batch base
    local ubatch=$(compute_ubatch)
    OVERRIDE_BATCH_SIZE="--batch-size 2048 --ubatch-size $ubatch"

    # ------------------------------------------------------------------
    # Profile selection (was multiple _preset_* functions)
    # ------------------------------------------------------------------
    local prof_ctx="" prof_checkpoint_min="" prof_ctx_checkpoints=""
    local prof_extra="" prof_ssd_disable="false"
    local prof_name="" prof_dspark=""
    local prof_slot_sim="" prof_no_checkpoint_end=""
    local prof_reasoning_budget="${LLAMA_REASONING_BUDGET}"
    local prof_cache_ram=""   # if non-empty, forces --cache-ram

    # Determine profile parameters
    if [[ "${is_ssm:-false}" == "true" ]]; then
        prof_name="ssm"
        if $is_strix_halo; then
            prof_ctx=262144
            prof_cache_ram=16384
        else
            prof_ctx=65536
            local vram_gb=${LLAMA_APU_VRAM_GB:-${GPU_BUDGET_GB:-4}}
            prof_cache_ram=$(( vram_gb * 1024 - 2048 ))
            (( prof_cache_ram < 1024 )) && prof_cache_ram=1024
        fi
        prof_extra="--no-context-shift --ctx-checkpoints 0"
        prof_ssd_disable="true"
        OVERRIDE_BATCH_SIZE="--batch-size 1024 --ubatch-size 512"
    elif $is_strix_halo && [[ "${is_moe:-false}" == "true" ]]; then
        if [[ $size_gb -gt 50 ]]; then
            prof_name="halo-moe-large"
            prof_ctx=131072
            prof_checkpoint_min=32768
            prof_ctx_checkpoints=2
            prof_ssd_disable="true"
        else
            prof_name="halo-moe-small"
            prof_ctx=196608
            prof_checkpoint_min=32768
            prof_ctx_checkpoints=2
            prof_ssd_disable="true"
        fi
    elif $is_strix_halo; then
        prof_name="halo-dense"
        prof_ctx=131072
        prof_checkpoint_min=8192
        prof_ctx_checkpoints=8
        prof_ssd_disable="true"
        prof_slot_sim="0.15"
        prof_no_checkpoint_end="true"
    elif [[ "${is_moe:-false}" == "true" ]]; then
        if [[ $size_gb -gt 18 ]]; then
            prof_name="std-moe-large"
            prof_ctx=65536
            prof_checkpoint_min=8192
            prof_ctx_checkpoints=8
        else
            prof_name="std-moe-small"
            prof_ctx=32768
            prof_checkpoint_min=8192
            prof_ctx_checkpoints=4
        fi
        prof_no_checkpoint_end="true"
    else
        if [[ $size_gb -gt 15 ]]; then
            prof_name="std-dense-large"
        else
            prof_name="std-dense"
        fi
        prof_ctx=32768
        prof_checkpoint_min=8192
        prof_ctx_checkpoints=4
        prof_no_checkpoint_end="true"
        prof_slot_sim="0.15"
    fi

    profile_name="$prof_name"

    # Apply context size (unless user forced it)
    [[ -z "$USER_CTX_SIZE" ]] && CTX_SIZE="${prof_ctx:-32768}"

    # Base server args (loaded mode, flash attention, etc.)
    EXTRA_SERVER_ARGS="--no-mmproj --load-mode ${LLAMA_LOAD_MODE} --flash-attn ${LLAMA_FLASH_ATTN}"

    # Checkpoint / context shift flags
    [[ -n "$prof_checkpoint_min" ]] && EXTRA_SERVER_ARGS+=" --checkpoint-min-step $prof_checkpoint_min"
    [[ -n "$prof_ctx_checkpoints" ]] && EXTRA_SERVER_ARGS+=" --ctx-checkpoints $prof_ctx_checkpoints"
    [[ "$prof_no_checkpoint_end" == "true" ]] && EXTRA_SERVER_ARGS+=" --no-checkpoint-near-end"
    [[ -n "$prof_slot_sim" ]] && EXTRA_SERVER_ARGS+=" --slot-prompt-similarity $prof_slot_sim"

    # Extra flags from profile
    [[ -n "$prof_extra" ]] && EXTRA_SERVER_ARGS+=" $prof_extra"

    # SSD cache (unless disabled by profile)
    _SSD_DISABLE="$prof_ssd_disable"
    if [[ "$prof_ssd_disable" != "true" ]]; then
        _apply_ssd_defaults   # populates SSD_* variables from central defaults
    fi

    # Reasoning defaults (every preset except maybe some special ones)
    _apply_reasoning_defaults

    # Override reasoning budget if profile wants a different one (currently all 2048)
    OVERRIDE_REASONING_BUDGET="$prof_reasoning_budget"

    # ------------------------------------------------------------------
    # Speculative decoding (mutually exclusive: DSpark / DFlash / MTP).
    # Each branch adds `-md <draft> --spec-type <type> --spec-draft-n-max N
    # --fit off` to EXTRA_SERVER_ARGS. Priority is DSpark (DeepSeek's own
    # Spark draft) > DFlash (block-diffusion, e.g. Laguna) > MTP (head
    # self-speculative; no separate draft file needed). n_max values are
    # tuned per-type and overridable via the LLAMA_SPEC_DRAFT_N_MAX_*
    # environment variables at the top of the file.
    #
    # Pre-detect DSpark and DFlash candidates BEFORE picking the winner
    # so compute_cache_ram() (called below) can size --cache-ram against
    # the draft's file size + KV cache cost. DSpark and DFlash are not
    # mutually exclusive at the file-system level (both may exist in
    # $MODEL_DIR); only the winner is actually loaded, but both must be
    # checked so we know the cost up front.
    # ------------------------------------------------------------------
    if [[ -z "${prof_dspark:-}" ]]; then
        prof_dspark=$(_detect_draft_model "$MODEL" "dspark")
    fi
    if [[ -z "${prof_dflash:-}" ]]; then
        prof_dflash=$(_detect_draft_model "$MODEL" "dflash")
    fi

    # DSpark (DeepSeek Spark) draft. Triggered when a matching
    # `<stem>-DSpark-{Q8_0,BF16,MXFP4}.gguf` exists in $MODEL_DIR next to
    # the main model. Currently used by DeepSeek-V4-Flash on halo-moe-large.
    # `n_max=4` is tuned for DeepSeek-V4-Flash: above this the draft starts
    # to spend cycles on tokens that get rejected, below it the target
    # spends more time verifying individual drafts. See _detect_draft_model
    # for the matching convention (case-insensitive `-DSpark-` suffix).
    if [[ -n "$prof_dspark" ]]; then
        log_info "DSpark speculative decoding enabled: $prof_dspark (n_max=$LLAMA_SPEC_DRAFT_N_MAX_DSPARK)"
        EXTRA_SERVER_ARGS+=" -md $prof_dspark --spec-type draft-dspark --spec-draft-n-max $LLAMA_SPEC_DRAFT_N_MAX_DSPARK --fit off"

    # MTP (Multi-Token Prediction) head self-speculative decoding. The MTP
    # head tensors (blk.N.nextn.{eh_proj,enorm,hnorm,shared_head_*,
    # embed_tokens}) live in the same GGUF as the main model, so no
    # separate `-md` is needed. Triggered when the GGUF header carries
    # `nextn_predict_layers > 0` (architectures: qwen3next / qwen35 /
    # qwen35moe / deepseek4 / deepseek2 / cohere2moe / bailingmoe2 /
    # deepseek32). `n_max=2` is Unsloth's recommendation for Qwen 3.6:
    # each MTP forward emits one token and the driver loops to draft a
    # small block before target verification.
    elif [[ "${is_mtp:-false}" == "true" ]]; then
        log_info "MTP speculative decoding enabled (n_max=$LLAMA_SPEC_DRAFT_N_MAX_MTP)"
        EXTRA_SERVER_ARGS+=" --spec-type draft-mtp --spec-draft-n-max $LLAMA_SPEC_DRAFT_N_MAX_MTP --fit off"

    # DFlash block-diffusion speculative decoding. Triggered when a matching
    # `<stem>-DFlash-*.gguf` exists in $MODEL_DIR. Currently used by
    # Laguna-S 2.1 (`Laguna-S-2.1-DFlash-BF16.gguf` next to
    # `Laguna-S-2.1-UD-Q4_K_XL-...`). `n_max=7` is empirical: DFlash drafts
    # in `block_size=16` chunks (server logs confirm) but observed draft
    # acceptance on Laguna is ~21% with mean accepted batch length ~2.5
    # tokens, so larger n_max wastes draft forward passes on tokens that
    # will be rejected; 7 sits just above the natural acceptance ceiling
    # and leaves headroom for occasional longer matches without inflating
    # per-batch overhead. prof_dflash is pre-detected above so the
    # cache-ram budget can account for its cost.
    elif [[ -n "$prof_dflash" ]]; then
        log_info "DFlash speculative decoding enabled: $prof_dflash (n_max=$LLAMA_SPEC_DRAFT_N_MAX_DFLASH)"
        EXTRA_SERVER_ARGS+=" -md $prof_dflash --spec-type draft-dflash --spec-draft-n-max $LLAMA_SPEC_DRAFT_N_MAX_DFLASH --fit off"
    fi

    : "${prof_cache_ram:=$(compute_cache_ram)}"
    EXTRA_SERVER_ARGS+=" --cache-ram $prof_cache_ram"

    # MoE expert residency / cpu-moe (after cache-ram to avoid overriding)
    _apply_moe_streaming "$size_gb" "$is_strix_halo" "$is_moe"

    log_info "Dynamic settings: KV=${KV_CACHE_TYPE_K}/${KV_CACHE_TYPE_V}, ubatch=${ubatch}, cache-ram=${prof_cache_ram} MiB"
    printf '%bAuto profile: %b%s%b (%sGB, MoE=%s, SSM=%s)%b\n' "$CYAN" "$GREEN" "$profile_name" "$NC" "$size_gb" "${is_moe:-false}" "${is_ssm:-false}" "$NC"
}

_scan_gguf_arch() {
    local gguf_path="$1"
    [[ ! -f "$gguf_path" || ! -r "$gguf_path" ]] && return 0
    local tmp=$(mktemp /tmp/llama-scan-XXXXXX)
    dd if="$gguf_path" of="$tmp" bs=16384 count=1 2>/dev/null || { rm -f "$tmp"; return 0; }
    if grep -q 'expert_count' "$tmp" 2>/dev/null; then is_moe=true; fi
    if grep -q 'ssm\.' "$tmp" 2>/dev/null && ! grep -q 'full_attention_interval' "$tmp" 2>/dev/null && [[ "${is_moe:-false}" != "true" ]]; then
        is_ssm=true
    fi
    # MTP head detection. `nextn_predict_layers` is the per-arch key written
    # by `LLM_KV_NEXTN_PREDICT_LAYERS` (e.g. `qwen35moe.nextn_predict_layers`)
    # and lives in the GGUF KV-metadata section -- always within the first
    # 16 KiB so `dd bs=16384 count=1` covers it. Architectures with native
    # MTP support include qwen3next / qwen35 / qwen35moe / deepseek4 /
    # deepseek2 / cohere2moe / bailingmoe2 / deepseek32. We do NOT also
    # grep for `nextn.eh_proj` tensor names because those live in the
    # tensor-info section after KV metadata and are often beyond the 16 KiB
    # window -- false negatives, not false positives, are the risk.
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

_apply_moe_streaming() {
    local size_gb="$1" strix="$2" is_moe="$3"
    if [[ "$is_moe" != "true" ]] || [[ ${GPU_BUDGET_BYTES:-0} -eq 0 ]]; then return; fi
    if [[ ${MODEL_BYTES:-0} -gt ${GPU_BUDGET_BYTES} ]]; then
        log_info "MoE expert weights pinned to CPU RAM (model ${size_gb}GB exceeds GPU budget ${GPU_BUDGET_GB}GB)"
        local total_mem=$(get_total_memory_bytes)
        if [[ ${MODEL_BYTES:-0} -lt $total_mem ]]; then
            EXTRA_SERVER_ARGS+=" --cpu-moe --load-mode none"
        else
            EXTRA_SERVER_ARGS+=" --cpu-moe"
        fi
    elif [[ ${MODEL_BYTES:-0} -gt $((GPU_BUDGET_BYTES * 95 / 100)) ]]; then
        if [[ "$strix" == "true" ]] || check_ac_power; then
            EXTRA_SERVER_ARGS+=" --moe-expert-residency"
        else
            log_warn "MoE expert residency skipped: standard tier on battery"
        fi
    fi
}

# Find an auxiliary draft model next to the main model. Auxiliary models
# follow the convention `<main-base>-<TAG>-<quant>.gguf` where TAG is the
# lowercased draft architecture name (e.g. "dspark" for DeepSeek Spark,
# "dflash" for Laguna DFlash block diffusion). Match is anchored end-to-
# end and case-insensitive so a draft for an unrelated model in a shared
# models dir cannot false-match. Drafts are not quant-matched to the main
# model -- e.g. `<base>-DSpark-Q8_0.gguf` and `<base>-DFlash-BF16.gguf` are
# both valid drafts for the same main model across quants. Returns the
# first matching file path, or empty string.
#
# Usage: _detect_draft_model "$MODEL" "<tag>"
_detect_draft_model() {
    local model_path="$1" tag_lc="$2"
    [[ -z "$tag_lc" ]] && { echo ""; return; }

    local base
    base=$(basename "$model_path")
    # Strip shard suffix "-NNNNN-of-NNNNN.gguf" if present
    if [[ "$base" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        base="${base%-${BASH_REMATCH[1]}-of-${BASH_REMATCH[2]}.gguf}"
    fi
    # Strip trailing quant segment and -UD marker so the stem matches the
    # draft's stem (drafts are not quant-matched to the main model).
    base="${base%-*}"
    [[ "$base" == *"-UD" ]] && base="${base%-UD}"
    [[ -z "$base" ]] && { echo ""; return; }

    local base_lc
    base_lc=$(echo "$base" | tr '[:upper:]' '[:lower:]')
    local f fb
    while IFS= read -r -d '' f; do
        fb=$(basename "$f")
        # Anchored regex: stem immediately followed by -<tag>- then any
        # quant-ish token and .gguf. bash globs / [[ -f ]] do not expand
        # patterns inside quoted variables and are case-sensitive, so we
        # lowercase both sides and use grep -qiE for the match.
        if echo "$fb" | grep -qiE "^${base_lc}-${tag_lc}-[A-Za-z0-9_]+\.gguf$"; then
            echo "$f" && return
        fi
    done < <(find "$MODEL_DIR" -maxdepth 1 -iname "*-${tag_lc}-*.gguf" -print0 2>/dev/null)
    echo ""
}

# -----------------------------------------------------------------------------
# Model discovery & download (unchanged)
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

# -----------------------------------------------------------------------------
# Hugging Face download helpers (unchanged from original)
# -----------------------------------------------------------------------------
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
# Backend environment setup (CRITICAL: previously missing apply_backend_env)
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

# Performance tuning helpers
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
# Usage
# -----------------------------------------------------------------------------
usage() {
    cat << USAGE
${BLUE}Llama.cpp Runner - Fully adaptive runtime detection

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
    --reasoning-budget N    Max thinking tokens per response (default: 2048)
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

# Variables that track user overrides for later use
USER_CTX_SIZE=""
USER_KV_CACHE_TYPE=""

# Initialise all variables used later to avoid 'unbound variable' errors
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
EXTRA_COMMON_ARGS=""      # <-- Fix: initialize to empty
EXTRA_SERVER_ARGS=""      # <-- Just in case
OVERRIDE_BATCH_SIZE=""    # <-- Initialize as well

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
        --no-ssd-cache) _SSD_DISABLE=true; shift ;;
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
        *) [[ -z "$MODEL" ]] && MODEL="$1"; shift ;;
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
# Profile summary (for --print-profile or startup)
# -----------------------------------------------------------------------------

# Parse a single `--flag VALUE` (or `-f VALUE`) occurrence from
# EXTRA_SERVER_ARGS. Echoes the first VALUE, or prints $2 if absent.
_extract_arg() {
    local flag="$1" fallback="$2"
    local val
    val=$(echo "$EXTRA_SERVER_ARGS" | sed -nE "s/.* ${flag} ([^ ]+).*/\\1/p" | head -1)
    [[ -z "$val" ]] && val="$fallback"
    echo "$val"
}

# Parse a single boolean `--flag` (no value) presence check from
# EXTRA_SERVER_ARGS. Echoes yes/no.
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
    load_mode=$(_extract_arg --load-mode "${LLAMA_LOAD_MODE}")
    flash_attn=$(_extract_arg --flash-attn "${LLAMA_FLASH_ATTN}")
    slot_sim=$(_extract_arg --slot-prompt-similarity "")
    spec_type=$(_extract_arg --spec-type "")
    spec_n_max=$(_extract_arg --spec-draft-n-max "")
    checkpoint_min=$(_extract_arg --checkpoint-min-step "")
    checkpoint_count=$(_extract_arg --ctx-checkpoints "")

    # Chat template: --chat-template-file value (or model default if unset).
    # Value is single-quoted in EXTRA_SERVER_ARGS, so we strip the quotes.
    local _ct
    _ct=$(echo "$EXTRA_SERVER_ARGS" | sed -nE "s/.*--chat-template-file '([^']+)'.*/\\1/p" | head -1)
    [[ -z "$_ct" ]] && chat_template="model default" || chat_template=$(basename "$_ct")

    # Draft model path (DSpark / DFlash) via -md. MTP self-speculative, no
    # draft file -- leave spec_draft empty so the display shows "self".
    spec_draft=$(echo "$EXTRA_SERVER_ARGS" | sed -nE 's/.* -md ([^ ]+).*/\1/p' | head -1)
    if [[ -n "$spec_draft" ]]; then
        spec_draft=$(basename "$spec_draft" .gguf)
    fi

    # MoE expert strategy. Order matters: --cpu-moe is most aggressive (all
    # experts on host RAM, --load-mode none), --moe-expert-residency is the
    # mmap'd middle path, neither means experts stay on GPU.
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

    # Spec decode summary. spec_type may be a comma-separated list (e.g.
    # "ngram-simple,draft-mtp"); display the highest-priority entry.
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
    printf '  %-28s %s\n' "Threads:"         "${THREADS}"
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

    # RAM budget breakdown. Shows exactly what was subtracted from the
    # GPU budget to land on the final --cache-ram value, so the user
    # can see the impact of each enabled parameter (SD draft, SSD, MTP,
    # n-gram caches, --np, etc.). Skip the section if there's nothing
    # to show (CPU-only backend or no budget).
    if [[ ${GPU_BUDGET_BYTES:-0} -gt 0 ]]; then
        compute_ram_budget_breakdown
        local budget_gib=$(awk -v b="$GPU_BUDGET_BYTES" 'BEGIN { printf "%.1f", b / 1073741824 }')
        printf '%b─── RAM Budget ─────────────────────────────────────────────%b\n' "$BLUE" "$NC"
        printf '  %-28s %s GiB\n' "GPU budget:" "$budget_gib"
        printf '  %-28s %s GiB\n' "Target model:" "$(awk -v b="$MODEL_BYTES" 'BEGIN { printf "%.1f", b / 1073741824 }')"
        local _label _bytes _mib
        while IFS='=' read -r _label _bytes; do
            [[ -z "$_label" || "$_label" == \#* ]] && continue
            [[ "$_bytes" =~ ^[0-9]+$ ]] || continue
            _mib=$(( _bytes / 1048576 ))
            # Only show meaningful entries (>= 64 MiB -- otherwise it's
            # rounding noise on a 6 GiB budget).
            [[ $_mib -lt 64 ]] && continue
            printf '  %-28s %s MiB\n' "  - ${_label}:" "$_mib"
        done <<< "$_RAM_BUDGET_BREAKDOWN"
        printf '  %-28s %s MiB\n' "Available for cache-ram:" "${cache_ram_val}"
        printf '%b──────────────────────────────────────────────────────────%b\n' "$BLUE" "$NC"
    fi
}

if $PRINT_PROFILE; then
    # Formatted human-readable summary first (matches what --server shows).
    print_profile_summary
    echo ""
    # Then variable dump for scripting. Output is parseable as `key=value`.
    cat <<PROFILE_EOF
CTX_SIZE=$CTX_SIZE
MODEL_PATH='$MODEL'
MODEL_NAME=$(basename "$MODEL" .gguf)
MODEL_BYTES=$MODEL_BYTES
GPU_LAYERS=$GPU_LAYERS
THREADS=$THREADS
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
apply_backend_env      # <-- Now defined, fixes the missing command error
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
MEMLOCK_LIMIT_KB=$(ulimit -l 2>/dev/null || echo 0)
[[ "$MEMLOCK_LIMIT_KB" == "unlimited" ]] && MEMLOCK_LIMIT_KB=0
if [[ $MODEL_BYTES -gt 0 && $MODEL_BYTES -gt $((MEMLOCK_LIMIT_KB * 1024)) ]]; then
    log_info "mlock disabled: model ($((MODEL_BYTES / 1048576)) MiB) larger than memlock limit ($((MEMLOCK_LIMIT_KB * 1024 / 1048576)) MiB)"
else
    COMMON_ARGS+=" --load-mode mlock"
fi
COMMON_ARGS+=" -c $CTX_SIZE --threads $THREADS --threads-batch $THREADS"
COMMON_ARGS+=" ${OVERRIDE_BATCH_SIZE} -ngl $GPU_LAYERS"
[[ -n "$OVERRIDE_UBATCH_SIZE" ]] && COMMON_ARGS=$(echo "$COMMON_ARGS" | sed -E 's/--ubatch-size [0-9]+/--ubatch-size '"$OVERRIDE_UBATCH_SIZE"'/')
COMMON_ARGS+=" --cache-type-k $KV_CACHE_TYPE_K --cache-type-v $KV_CACHE_TYPE_V"
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

# Context shifting toggle
if [[ "${ENABLE_CONTEXT_SHIFT:-1}" == "0" ]]; then
    SERVER_ARGS+=" --no-context-shift"
    log_info "Context shifting DISABLED"
else
    log_info "Context shifting ENABLED"
fi

# Preserve reasoning template kwarg
[[ "$PRESERVE_REASONING" == "true" ]] && SERVER_ARGS+=" --reasoning-preserve --chat-template-kwargs '{\"preserve_thinking\":true}'"

# SSD cache (if enabled)
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
