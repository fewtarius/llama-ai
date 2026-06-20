# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 55239.6ms | 15489 | 54745.3ms | 1.01x | 3.6ms | 3.5ms | 25.2ms | miss |
| medium | 5237 | 10951.3ms | 5237 | 10839.9ms | 1.01x | 2.1ms | 2.1ms | 18.8ms | miss |
| small | 1145 | 2445.2ms | 1145 | 2035.6ms | 1.12x | 2.1ms | 1.8ms | 16.9ms | miss |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
