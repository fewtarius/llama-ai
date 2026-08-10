#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Regression tests for scripts/lib-discover-models.sh
#
# Verifies the model discovery used by scripts/benchmark.sh:
#   * Walks the directory dynamically (no hard-coded list)
#   * Skips DFlash draft artifacts (case-insensitive)
#   * Skips non-first shards of multipart models
#   * Keeps the first shard of complete multipart sets
#   * Skips incomplete multipart sets with a warning
#   * Does not recurse into subdirectories
#   * Emits results in sorted order with the `<name>:` sentinel
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$PROJECT_ROOT/scripts/lib-discover-models.sh"
[[ -f "$LIB" ]] || { echo "FAIL: $LIB not found" >&2; exit 1; }
# shellcheck source=../scripts/lib-discover-models.sh
source "$LIB"

TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

fail() {
    echo "FAIL [$1]:" >&2
    shift
    for line in "$@"; do
        echo "  $line" >&2
    done
    exit 1
}

assert_eq() {
    local actual="$1" expected="$2" label="$3"
    if [[ "$actual" != "$expected" ]]; then
        fail "$label" \
            "expected:" "  $(printf '%q' "$expected")" \
            "actual:"   "  $(printf '%q' "$actual")"
    fi
}

# mkfile PATH SIZE_BYTES  - create a sparse file of the given size
mkfile() {
    local path="$1" size="$2"
    dd if=/dev/zero of="$path" bs=1 count=0 seek="$size" 2>/dev/null
}

# list_models DIR  - capture discover_models output as a single string
list_models() {
    local dir="$1"
    local -a arr
    mapfile -t arr < <(discover_models "$dir")
    printf '%s\n' "${arr[@]}"
}

# ─── Test 1: empty directory returns nothing ─────────────────────────────
out=$(list_models "$TEST_DIR/empty")
assert_eq "$out" "" "empty dir returns no models"

# ─── Test 2: single-file models are all kept, sorted alphabetically ─────
mkdir -p "$TEST_DIR/single"
touch "$TEST_DIR/single/A-B-C.gguf" \
      "$TEST_DIR/single/Z-Y-X.gguf" \
      "$TEST_DIR/single/M-N-O.gguf"
out=$(list_models "$TEST_DIR/single")
assert_eq "$out" "A-B-C.gguf:
M-N-O.gguf:
Z-Y-X.gguf:" "single files sorted alphabetically"

# ─── Test 3: DFlash artifacts are skipped (case-insensitive) ────────────
mkdir -p "$TEST_DIR/dflash"
touch "$TEST_DIR/dflash/Model-A-UD-Q4_K_M.gguf" \
      "$TEST_DIR/dflash/Model-A-UD-dflash-Q8_0.gguf" \
      "$TEST_DIR/dflash/Model-B-DFlash-BF16.gguf" \
      "$TEST_DIR/dflash/Model-C-dFlash-Q4_K.gguf"
# A non-DFlash file containing the word 'dflash' but NOT as `-dflash-`
# must NOT be excluded. The convention is `<base>-dflash-<quant>.gguf`.
touch "$TEST_DIR/dflash/Model-D-dflashmodel-Q4_K.gguf"
out=$(list_models "$TEST_DIR/dflash")
assert_eq "$out" "Model-A-UD-Q4_K_M.gguf:
Model-D-dflashmodel-Q4_K.gguf:" "DFlash drafts filtered (case-insensitive, anchored)"

# ─── Test 4: only the first shard of a complete multipart set is kept ───
mkdir -p "$TEST_DIR/split-complete"
mkfile "$TEST_DIR/split-complete/Model-A-UD-Q4_K_XL-00001-of-00003.gguf" 16
mkfile "$TEST_DIR/split-complete/Model-A-UD-Q4_K_XL-00002-of-00003.gguf" 1024
mkfile "$TEST_DIR/split-complete/Model-A-UD-Q4_K_XL-00003-of-00003.gguf" 512
mkfile "$TEST_DIR/split-complete/Model-B-UD-Q8_K_XL-00001-of-00002.gguf" 16
mkfile "$TEST_DIR/split-complete/Model-B-UD-Q8_K_XL-00002-of-00002.gguf" 1024
mkfile "$TEST_DIR/split-complete/Model-C-UD-Q6_K-00001-of-00004.gguf" 16
mkfile "$TEST_DIR/split-complete/Model-C-UD-Q6_K-00002-of-00004.gguf" 1024
mkfile "$TEST_DIR/split-complete/Model-C-UD-Q6_K-00003-of-00004.gguf" 1024
mkfile "$TEST_DIR/split-complete/Model-C-UD-Q6_K-00004-of-00004.gguf" 1024
out=$(list_models "$TEST_DIR/split-complete")
assert_eq "$out" "Model-A-UD-Q4_K_XL-00001-of-00003.gguf:
Model-B-UD-Q8_K_XL-00001-of-00002.gguf:
Model-C-UD-Q6_K-00001-of-00004.gguf:" "only first shard of complete sets"

# ─── Test 5: incomplete multipart sets are skipped with a warning ───────
mkdir -p "$TEST_DIR/split-incomplete"
# X: missing 00003
mkfile "$TEST_DIR/split-incomplete/Model-X-UD-Q4_K_M-00001-of-00003.gguf" 16
mkfile "$TEST_DIR/split-incomplete/Model-X-UD-Q4_K_M-00002-of-00003.gguf" 1024
# Y: missing 00002
mkfile "$TEST_DIR/split-incomplete/Model-Y-UD-Q4_K_M-00001-of-00002.gguf" 16
# Z: 00002 is 0 bytes (interrupted download) - must be treated as missing
mkfile "$TEST_DIR/split-incomplete/Model-Z-UD-Q4_K_M-00001-of-00003.gguf" 16
mkfile "$TEST_DIR/split-incomplete/Model-Z-UD-Q4_K_M-00002-of-00003.gguf" 0
mkfile "$TEST_DIR/split-incomplete/Model-Z-UD-Q4_K_M-00003-of-00003.gguf" 1024

warn_err=$(discover_models "$TEST_DIR/split-incomplete" 2>&1 >/dev/null || true)
for expect in "Model-X-UD-Q4_K_M-00001-of-00003" "Model-Y-UD-Q4_K_M-00001-of-00002" "Model-Z-UD-Q4_K_M-00001-of-00003"; do
    if [[ "$warn_err" != *"$expect"* ]]; then
        fail "incomplete-set warning" \
            "expected warning for $expect" \
            "got: $warn_err"
    fi
done
# X: 00002 present, 00003 missing -> warning should mention '00003'
if [[ "$warn_err" != *"Model-X-UD-Q4_K_M-00001-of-00003.gguf (missing shards: 00003)"* ]]; then
    fail "missing-shard list" \
        "expected 'missing shards: 00003' for X" \
        "got: $warn_err"
fi
# Y: 00001 present, 00002 missing
if [[ "$warn_err" != *"Model-Y-UD-Q4_K_M-00001-of-00002.gguf (missing shards: 00002)"* ]]; then
    fail "missing-shard list" \
        "expected 'missing shards: 00002' for Y" \
        "got: $warn_err"
fi
# Z: 00002 zero-bytes -> reported as missing
if [[ "$warn_err" != *"Model-Z-UD-Q4_K_M-00001-of-00003.gguf (missing shards: 00002)"* ]]; then
    fail "zero-byte shard treated as missing" \
        "expected 'missing shards: 00002' for Z" \
        "got: $warn_err"
fi
out=$(list_models "$TEST_DIR/split-incomplete")
assert_eq "$out" "" "incomplete multipart sets skipped"

# ─── Test 6: subdirectories are not recursed ─────────────────────────────
mkdir -p "$TEST_DIR/skip-subdirs/disable"
touch "$TEST_DIR/skip-subdirs/Top-Model.gguf" \
      "$TEST_DIR/skip-subdirs/disable/Hidden-Model.gguf" \
      "$TEST_DIR/skip-subdirs/disable/Hidden-DFlash-BF16.gguf"
out=$(list_models "$TEST_DIR/skip-subdirs")
assert_eq "$out" "Top-Model.gguf:" "subdirectories are not recursed"

# ─── Test 7: combined real-world scenario ────────────────────────────────
# Mirrors the current /home/deck/llama-ai/models directory layout.
mkdir -p "$TEST_DIR/realistic"
# Single-file models
mkfile "$TEST_DIR/realistic/GLM-4.7-Flash-UD-Q8_K_XL.gguf" 1024
mkfile "$TEST_DIR/realistic/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf" 1024
mkfile "$TEST_DIR/realistic/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" 1024
mkfile "$TEST_DIR/realistic/gpt-oss-20b-UD-Q6_K_XL.gguf" 1024
# DFlash artifact (should be skipped)
mkfile "$TEST_DIR/realistic/Laguna-S-2.1-DFlash-BF16.gguf" 1024
# Multipart complete
for i in 1 2 3 4; do
    n=$(printf '%05d' "$i")
    mkfile "$TEST_DIR/realistic/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-${n}-of-00004.gguf" 1024
done
# Multipart incomplete (00003 missing)
mkfile "$TEST_DIR/realistic/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf" 1024
mkfile "$TEST_DIR/realistic/MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf" 1024
# A 2-shard set, complete
mkfile "$TEST_DIR/realistic/Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002.gguf" 1024
mkfile "$TEST_DIR/realistic/Qwen3-235B-A22B-Thinking-2507-IQ2_M-00002-of-00002.gguf" 1024
# `disable/` subdir should not be entered
mkdir -p "$TEST_DIR/realistic/disable"
mkfile "$TEST_DIR/realistic/disable/laguna-s-2.1-dflash-Q8_0.gguf" 1024
mkfile "$TEST_DIR/realistic/disable/dspark-DeepSeek-V4-Flash-0731-BF16.gguf" 1024

out=$(list_models "$TEST_DIR/realistic")
expected="DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf:
GLM-4.7-Flash-UD-Q8_K_XL.gguf:
Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002.gguf:
Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf:
dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf:
gpt-oss-20b-UD-Q6_K_XL.gguf:"
assert_eq "$out" "$expected" "realistic mixed scenario"

# ─── Test 8: non-existent directory is handled gracefully ────────────────
out=$(list_models "$TEST_DIR/does-not-exist")
assert_eq "$out" "" "non-existent directory returns nothing"

# ─── Test 9: caller can override MODEL_DIR default via arg ───────────────
# (sanity check that the optional arg is honored)
mkdir -p "$TEST_DIR/alt"
touch "$TEST_DIR/alt/Alt-Model.gguf"
out=$(list_models "$TEST_DIR/alt")
assert_eq "$out" "Alt-Model.gguf:" "MODEL_DIR arg honored"

echo "discover_models tests passed"
