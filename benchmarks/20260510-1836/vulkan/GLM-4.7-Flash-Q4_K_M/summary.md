# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 254908.1ms | 15489 | 4130.8ms | 17.97x | 16.5ms | 1032.7ms | 81.8ms | ssd_cold |
| medium | 5237 | 55883.9ms | 5237 | 1754.3ms | 7.39x | 10.7ms | 438.6ms | 51.1ms | ssd_cold |
| small | 1145 | 9743.5ms | 1145 | 462.9ms | 2.7x | 8.5ms | 115.7ms | 38.3ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
