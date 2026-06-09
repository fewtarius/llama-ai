# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 520443.9ms | 17347 | 3275.4ms | 28.61x | 30.0ms | 818.9ms | 117.8ms | ssd_cold |
| medium | 6083 | 104071.5ms | 6083 | 1918.6ms | 7.3x | 17.1ms | 479.7ms | 109.4ms | ssd_cold |
| small | 1413 | 18895.1ms | 1413 | 1661.2ms | 2.13x | 13.4ms | 415.3ms | 105.5ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
