# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 439068.3ms | 15489 | 2680.1ms | 20.49x | 28.3ms | 670.0ms | 161.9ms | ssd_cold |
| medium | 5237 | 74375.3ms | 5237 | 1022.5ms | 7.87x | 14.2ms | 255.6ms | 77.7ms | ssd_cold |
| small | 1145 | 10285.0ms | 1145 | 350.1ms | 2.64x | 9.0ms | 87.5ms | 44.5ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
