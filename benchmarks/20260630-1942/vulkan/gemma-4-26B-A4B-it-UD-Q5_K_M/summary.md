# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 23641.2ms | 17347 | 165.3ms | 9.63x | 1.4ms | 41.3ms | 65.3ms | ssd_warm |
| medium | 6083 | 5898.7ms | 6083 | 106.2ms | 3.96x | 1.0ms | 26.5ms | 49.1ms | ssd_warm |
| small | 1413 | 1529.1ms | 1413 | 91.8ms | 2.3x | 1.1ms | 23.0ms | 41.7ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
