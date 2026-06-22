# GLM-4.7-Flash-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 65165.4ms | 15489 | 2367.8ms | 8.73x | 4.2ms | 591.9ms | 38.0ms | ssd_cold |
| medium | 5237 | 13208.4ms | 5237 | 1026.8ms | 3.21x | 2.5ms | 256.7ms | 32.3ms | ssd_cold |
| small | 1145 | 3020.0ms | 1145 | 250.5ms | 1.69x | 2.6ms | 62.6ms | 32.0ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
