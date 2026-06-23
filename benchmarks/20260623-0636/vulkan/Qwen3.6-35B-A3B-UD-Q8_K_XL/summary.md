# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 18261.4ms | 15721 | 1204.7ms | 4.61x | 1.2ms | 301.2ms | 25.9ms | ssd_cold |
| medium | 5409 | 6299.9ms | 5409 | 657.7ms | 2.26x | 1.2ms | 164.4ms | 24.1ms | ssd_cold |
| small | 1243 | 1880.2ms | 1243 | 328.3ms | 1.46x | 1.5ms | 82.1ms | 26.8ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
