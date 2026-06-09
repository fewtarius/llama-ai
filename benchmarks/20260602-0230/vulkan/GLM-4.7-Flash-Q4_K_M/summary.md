# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 0 | 0ms | 15489 | 93555.6ms | 0x | 0ms | 359.8ms | 0ms | ssd_cold |
| medium | 5237 | 357298.4ms | 5237 | 1870.7ms | 17.71x | 68.2ms | 467.7ms | 149.9ms | ssd_cold |
| small | 1145 | 28206.8ms | 1145 | 586.7ms | 3.3x | 24.6ms | 146.7ms | 89.5ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
