#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Sourceable model-discovery helper for scripts/benchmark.sh.
#
# Walks a directory of GGUF files and emits one line per benchmarkable model
# in the form `<name>:` (the trailing colon is the separator consumed by the
# `IFS=':' read -r model extra_flags` parsing in the benchmark loop).
#
# Filters applied (in this order):
#   1. Speculative-decoding draft artifacts (`*-dflash-*`, `dspark-*`,
#      and `mtp-*` sidecar prefixes, case-insensitive) are skipped -
#      these are draft weights for parent-context decoder architectures,
#      not standalone target models.
#   2. Non-first shards of split GGUFs (`-NNNNN-of-NNNNN.gguf` where the
#      leading NNNNN is not `00001`) are skipped - llama.cpp auto-loads
#      subsequent shards when the first is passed to `-m`.
#   3. First shards of incomplete multipart sets are skipped with a
#      warning to avoid `tensor ... not within file bounds` crashes on
#      half-downloaded models.
#
# Single-file models (no split suffix) and complete multipart sets are
# emitted in alphabetical order. Subdirectories (e.g. `models/disable/`)
# are not recursed into.
#
# Usage:
#   source scripts/lib-discover-models.sh
#   discover_models [MODEL_DIR]            # default: $MODEL_DIR from caller
# =============================================================================

set -euo pipefail

# discover_models [MODEL_DIR]
#
# Emits benchmarkable models found under MODEL_DIR (one `name:` per line,
# sorted alphabetically). MODEL_DIR defaults to the caller's $MODEL_DIR.
# Returns 0 always; emits nothing on an empty/missing directory.
discover_models() {
    local model_dir="${1:-${MODEL_DIR:-}}"
    if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
        return 0
    fi

    local -a found=()
    local f
    while IFS= read -r -d '' f; do
        local name
        name=$(basename "$f")

        # Speculative-decoding draft filter. Three conventions in the wild:
        #   * `-dflash-` infix (e.g. `Laguna-S-2.1-dflash-BF16.gguf`)
        #     - target-arch-aware DFlash drafts (see
        #     https://github.com/fewtarius/CachyLLama/blob/master/docs/dflash.md)
        #   * `dspark-` prefix OR `*-dspark-*` infix (e.g. `dspark-DeepSeek-V4-Flash-Q8_0.gguf`
        #     or `DeepSeek-V4-Flash-0731-DSpark-Q8_0.gguf`)
        #     - DeepSeek's DFlash draft model; loaded as `-md` for the
        #     target, fails standalone with `dflash requires ctx_other`
        #   * `mtp-` prefix (e.g. `mtp-Qwen3-27B-Q4_K_M.gguf`)
        #     - Multi-Token Prediction sidecar from the GGUF Naming spec
        # Match case-insensitively. The substrings "dflash", "dspark", "mtp-"
        # are specific enough to avoid false positives on legitimate model names.
        local name_lc="${name,,}"
        if [[ "$name_lc" == *dflash* || "$name_lc" == *dspark* || "$name_lc" == mtp-* ]]; then
            continue
        fi

        # Split-shard filter. The Hugging Face split convention is
        # `<base>-NNNNN-of-NNNNN.gguf`. The first shard is the entry point
        # and the rest are auto-loaded by llama.cpp when its path is passed
        # to `-m`. We keep only `00001-of-NNNNN` and drop the rest.
        if [[ "$name" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
            local shard="${BASH_REMATCH[1]}" total="${BASH_REMATCH[2]}"
            if [[ "$shard" != "00001" ]]; then
                continue
            fi

            # Incomplete-set guard. A half-downloaded multipart model would
            # crash the server with `tensor ... not within file bounds`
            # partway through loading; warn and skip so the rest of the
            # benchmark run is not blocked by one bad download.
            local base="${name%-${shard}-of-${total}.gguf}"
            local missing=()
            local i n shard_path
            # 10# prefix forces base-10 (otherwise `00004` is octal).
            for ((i = 1; i <= 10#$total; i++)); do
                printf -v n '%05d' "$i"
                shard_path="${model_dir}/${base}-${n}-of-${total}.gguf"
                # -s: exists AND non-empty. An interrupted download often
                # leaves a 0-byte file in place.
                if [[ ! -s "$shard_path" ]]; then
                    missing+=("$n")
                fi
            done
            if [[ ${#missing[@]} -gt 0 ]]; then
                printf '[WARN] discover_models: skipping %s (missing shards: %s)\n' \
                    "$name" "${missing[*]}" >&2 || true
                continue
            fi
        fi

        found+=("$name:")
    done < <(find "$model_dir" -maxdepth 1 -type f -name "*.gguf" -print0 2>/dev/null)

    if [[ ${#found[@]} -eq 0 ]]; then
        return 0
    fi
    # Sort for deterministic ordering across runs/filesystems.
    printf '%s\n' "${found[@]}" | LC_ALL=C sort
}
