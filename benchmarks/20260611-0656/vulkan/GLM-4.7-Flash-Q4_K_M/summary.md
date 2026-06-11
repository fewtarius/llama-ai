# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 467588.2ms | 15489 | 2686.1ms | 19.95x | 30.2ms | 671.5ms | 175.0ms | ssd_cold |
| medium | 5237 | 74198.4ms | 5237 | 1020.0ms | 7.24x | 14.2ms | 255.0ms | 82.8ms | ssd_cold |
| small | 1145 | 9737.0ms | 1145 | 343.5ms | 2.42x | 8.5ms | 85.9ms | 49.5ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
