#!/bin/bash
# bisect-echo-test.sh - Tests the "criss-cross applesauce" reproducer
#
# Usage:
#   bash bisect-echo-test.sh                    # uses default model (Qwen3.6)
#   LLAMA_MODEL=OtherModel bash bisect-echo-test.sh
#
# Exit codes:
#   0 = GOOD (model responds with "criss-cross applesauce" or close)
#   1 = BAD  (model fails to respond, loops, or generates >1000 decode tokens)
#   2 = TIMEOUT / setup error
#
# The test ALWAYS rebuilds CachyLLama first to ensure the binary matches
# the currently checked-out commit.

set -u

# Project root = parent of the directory containing this script.
# Works regardless of where the llama-ai repo is checked out.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$_SCRIPT_DIR/.." && pwd)"
CACHY_DIR="$PROJECT_DIR/CachyLLama"
BINARY="$PROJECT_DIR/src/cachy-llama-vulkan/build/bin/llama-server"
cd "$PROJECT_DIR" || exit 2

LLAMA_MODEL="${LLAMA_MODEL:-Qwen3.6-35B-A3B-UD-Q8_K_XL}"
SERVER_LOG="/tmp/bisect-echo-server.log"
CLIO_LOG="/tmp/bisect-echo-clio.log"
SERVER_PID_FILE="/tmp/bisect-echo-server.pid"

# Clean up any leftover server
kill_existing() {
    if [ -f "$SERVER_PID_FILE" ]; then
        local oldpid
        oldpid=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
        if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
            kill -9 "$oldpid" 2>/dev/null || true
            wait "$oldpid" 2>/dev/null || true
        fi
        rm -f "$SERVER_PID_FILE"
    fi
    pkill -9 -f llama-server 2>/dev/null || true
    sleep 2
    fuser -k -n tcp 9090 2>/dev/null || true
    sleep 1
}

cleanup() { kill_existing; }
trap cleanup EXIT

CACHY_HEAD=$(git -C "$CACHY_DIR" rev-parse --short HEAD)
CACHY_DATE=$(git -C "$CACHY_DIR" log -1 --format=%ai)
LLAMAAI_HEAD=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)

echo "=== Bisect echo test ==="
echo "Model:        $LLAMA_MODEL"
echo "llama-ai:     $LLAMAAI_HEAD"
echo "CachyLLama:   $CACHY_HEAD ($CACHY_DATE)"
echo "Binary mtime: $(stat -c %y "$BINARY" 2>/dev/null || echo 'no binary')"

# Step 1: rebuild CachyLLama to match current commit
echo
echo "--- Rebuilding CachyLLama ---"
if ! ./scripts/rebuild.sh --clean > /tmp/bisect-echo-rebuild.log 2>&1; then
    echo "ERROR: rebuild failed"
    tail -30 /tmp/bisect-echo-rebuild.log
    exit 2
fi
echo "Rebuild OK ($(grep -E '\[INFO\] \[OK\]|build complete' /tmp/bisect-echo-rebuild.log | tail -2 | tr '\n' ' '))"

# Step 2: clean up any leftover server
echo
echo "--- Cleaning up existing server ---"
kill_existing

# Step 3: start server via llama-run.sh.
# stdbuf forces line-buffering on the launcher so the log file is
# useful for post-mortem. We don't grep the log for readiness because
# llama-server's stdout inherits block buffering when redirected.
echo
echo "--- Starting server via llama-run.sh ---"
stdbuf -oL -eL ./llama-run.sh --server "$LLAMA_MODEL" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
echo "Server PID: $SERVER_PID"

# Wait for server to be ready - poll /v1/models for the loaded model.
# /v1/models only returns the model entry after init completes, so
# this is the correct readiness signal.
READY=0
for i in $(seq 1 180); do
    # /v1/models returns {"object":"list","data":[{...}]} once the model is loaded.
    # The id field is the full .gguf path, so match the basename without the .gguf suffix.
    if curl -s --max-time 2 "http://localhost:9090/v1/models" 2>/dev/null \
        | grep -q "\"id\":\".*${LLAMA_MODEL}"; then
        echo "Server ready after ${i}s (model listed in /v1/models)"
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server did not become ready in 180s"
    echo "--- Last 30 lines of server log: ---"
    tail -30 "$SERVER_LOG"
    echo "--- /v1/models response: ---"
    curl -s --max-time 2 "http://localhost:9090/v1/models" 2>&1
    exit 2
fi

# Step 4: run the test prompt (echo test).
echo
echo "--- Running test prompt ---"
START_TIME=$(date +%s)
echo "Repeat these exact words: criss-cross applesauce" | timeout 180 clio --model llama.cpp/local_model --exit > "$CLIO_LOG" 2>&1
CLIO_EXIT=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Step 5: report.
echo
echo "--- Result (elapsed: ${ELAPSED}s, exit=$CLIO_EXIT) ---"
# Show just the first 5 + last 5 [RESPONSE/TERMINAL/...] lines to keep log readable
grep -E "^\[(RESPONSE|TERMINAL|THINKING|FILE|OUTPUT|TODO|TASK|MEMORY)\]" "$CLIO_LOG" | head -10
echo "..."
grep -E "^\[(RESPONSE|TERMINAL|THINKING|FILE|OUTPUT|TODO|TASK|MEMORY)\]" "$CLIO_LOG" | tail -5

# Verdict: must contain "criss-cross applesauce" in response.
# Also count tokens via the eval timing line in the server log.
if grep -q "criss-cross applesauce" "$CLIO_LOG"; then
    DECODE_TOKENS=$(grep -E "print_timing.*eval time" "$SERVER_LOG" 2>/dev/null | tail -1 | \
        grep -oE "/ +[0-9]+ tokens" | head -1 | grep -oE "[0-9]+" || echo "0")
    if [ "${DECODE_TOKENS:-0}" -gt 1000 ] 2>/dev/null; then
        echo
        echo "VERDICT: BAD (response found but used $DECODE_TOKENS decode tokens - likely loop)"
        exit 1
    else
        echo
        echo "VERDICT: GOOD (response found, $DECODE_TOKENS decode tokens)"
        exit 0
    fi
elif grep -qE "ERROR|EXCEPTION|decode error" "$CLIO_LOG"; then
    echo
    echo "VERDICT: BAD (error in clio log)"
    exit 1
elif [ "$CLIO_EXIT" -ne 0 ]; then
    echo
    echo "VERDICT: BAD (clio exit=$CLIO_EXIT, possible timeout)"
    exit 1
else
    echo
    echo "VERDICT: BAD (no expected response found)"
    exit 1
fi
