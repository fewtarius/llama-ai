# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 125117.4ms | 15721 | 1061.7ms | 15.85x | 8.0ms | 265.4ms | 52.7ms | ssd_cold |
| medium | 5409 | 39052.2ms | 5409 | 630.7ms | 6.35x | 7.2ms | 157.7ms | 48.3ms | ssd_cold |
| small | 1243 | 8762.5ms | 1243 | 436.0ms | 2.25x | 7.0ms | 109.0ms | 46.8ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
