# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 59602.6ms | 15489 | 1590.0ms | 12.61x | 3.8ms | 397.5ms | 26.5ms | ssd_cold |
| medium | 5237 | 11812.2ms | 5237 | 531.6ms | 4.71x | 2.3ms | 132.9ms | 19.5ms | ssd_cold |
| small | 1145 | 2083.1ms | 1145 | 154.9ms | 1.84x | 1.8ms | 38.7ms | 16.7ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
