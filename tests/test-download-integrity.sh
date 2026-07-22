#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT

set_file_size() {
    local path="$1"
    local size="$2"
    dd if=/dev/zero of="$path" bs=1 count=0 seek="$size" 2>/dev/null
}

file_size() {
    local path="$1"
    stat -c %s "$path" 2>/dev/null || stat -f %z "$path"
}

setup_case() {
    local case_dir="$1"

    mkdir -p "$case_dir/scripts" "$case_dir/models" "$case_dir/bin"
    cp "$PROJECT_ROOT/llama-run.sh" "$case_dir/llama-run.sh"

    cat > "$case_dir/scripts/detect-gpu.sh" <<'EOF'
#!/bin/bash
LLAMA_THREADS=1
EOF

    cat > "$case_dir/bin/curl" <<'EOF'
#!/bin/bash
printf '%s\n' "$HF_REPO_JSON"
EOF

    cat > "$case_dir/bin/hf" <<'EOF'
#!/bin/bash
set -euo pipefail

filename="$3"
target_dir="$5"
mkdir -p "$target_dir/$(dirname "$filename")"
remote_size=$(printf '%s' "$HF_REPO_JSON" | jq -r --arg filename "$filename" \
    '.siblings[] | select(.rfilename == $filename) | .size')
dd if=/dev/zero of="$target_dir/$filename" bs=1 count=0 \
    seek="${HF_DOWNLOAD_SIZE:-$remote_size}" 2>/dev/null
printf '%s\n' "$filename" >> "$HF_MOCK_LOG"
EOF

    chmod +x "$case_dir/llama-run.sh" "$case_dir/bin/curl" "$case_dir/bin/hf"
}

run_download() {
    local case_dir="$1"
    local repo_json="$2"
    local download_size="${3:-}"

    PATH="$case_dir/bin:$PATH" \
        HF_REPO_JSON="$repo_json" \
        HF_DOWNLOAD_SIZE="$download_size" \
        HF_MOCK_LOG="$case_dir/hf.log" \
        "$case_dir/llama-run.sh" --download owner/repo --quant Q4_K_M
}

single_repo='{"siblings":[{"rfilename":"Example-Q4_K_M.gguf","size":16}]}'

case_dir="$TEST_ROOT/valid-cache"
setup_case "$case_dir"
set_file_size "$case_dir/models/Example-Q4_K_M.gguf" 16
output=$(run_download "$case_dir" "$single_repo")
grep -q 'Already exists: Example-Q4_K_M.gguf' <<< "$output"
grep -q 'All files already cached' <<< "$output"
[[ ! -e "$case_dir/hf.log" ]]

split_repo='{"siblings":[
    {"rfilename":"Example-Q4_K_M-00001-of-00003.gguf","size":4},
    {"rfilename":"Example-Q4_K_M-00002-of-00003.gguf","size":16},
    {"rfilename":"Example-Q4_K_M-00003-of-00003.gguf","size":8}
]}'

case_dir="$TEST_ROOT/truncated-split-cache"
setup_case "$case_dir"
set_file_size "$case_dir/models/Example-Q4_K_M-00001-of-00003.gguf" 4
set_file_size "$case_dir/models/Example-Q4_K_M-00002-of-00003.gguf" 5
set_file_size "$case_dir/models/Example-Q4_K_M-00003-of-00003.gguf" 8
output=$(run_download "$case_dir" "$split_repo")
grep -q 'Cached file has wrong size: Example-Q4_K_M-00002-of-00003.gguf' <<< "$output"
grep -q 'All files downloaded successfully' <<< "$output"
[[ "$(file_size "$case_dir/models/Example-Q4_K_M-00001-of-00003.gguf")" == 4 ]]
[[ "$(file_size "$case_dir/models/Example-Q4_K_M-00002-of-00003.gguf")" == 16 ]]
[[ "$(file_size "$case_dir/models/Example-Q4_K_M-00003-of-00003.gguf")" == 8 ]]
[[ "$(wc -l < "$case_dir/hf.log")" == 1 ]]
grep -qx 'Example-Q4_K_M-00002-of-00003.gguf' "$case_dir/hf.log"

case_dir="$TEST_ROOT/bad-download"
setup_case "$case_dir"
if output=$(run_download "$case_dir" "$single_repo" 7 2>&1); then
    echo 'download unexpectedly accepted a file with the wrong size' >&2
    exit 1
fi
grep -q 'Invalid size: Example-Q4_K_M.gguf' <<< "$output"

echo 'download integrity tests passed'
