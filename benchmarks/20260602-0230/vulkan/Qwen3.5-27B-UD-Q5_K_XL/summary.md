# Qwen3.5-27B-UD-Q5_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 779802.3ms | 15721 | 6005.6ms | 11.88x | 49.6ms | 1501.4ms | 505.3ms | ssd_cold |
| medium | 5409 | 241283.2ms | 5409 | 3997.9ms | 4.59x | 44.6ms | 999.5ms | 483.0ms | ssd_cold |
| small | 1243 | 54619.8ms | 1243 | 1999.9ms | 1.84x | 43.9ms | 500.0ms | 475.9ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
