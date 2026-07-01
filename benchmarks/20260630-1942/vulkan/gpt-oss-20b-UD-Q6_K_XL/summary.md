# gpt-oss-20b-UD-Q6_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15380 | 12749.1ms | 15380 | 165.6ms | 8.28x | 0.8ms | 41.4ms | 34.3ms | ssd_warm |
| medium | 5250 | 3773.9ms | 5250 | 61.9ms | 3.48x | 0.7ms | 15.5ms | 19.3ms | ssd_warm |
| small | 1205 | 1036.6ms | 1205 | 36.8ms | 1.74x | 0.9ms | 9.2ms | 15.0ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
