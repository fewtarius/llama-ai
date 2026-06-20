# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 31008.9ms | 17347 | 539.5ms | 8.7x | 1.8ms | 134.9ms | 26.5ms | ssd_cold |
| medium | 6083 | 9285.8ms | 6083 | 321.9ms | 3.58x | 1.5ms | 80.5ms | 24.2ms | ssd_cold |
| small | 1413 | 2205.6ms | 1413 | 304.4ms | 1.6x | 1.6ms | 76.1ms | 23.8ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
