# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 113358.9ms | 17347 | 806.8ms | 13.5x | 6.5ms | 201.7ms | 62.2ms | ssd_cold |
| medium | 6083 | 33980.2ms | 6083 | 514.5ms | 5.32x | 5.6ms | 128.6ms | 57.5ms | ssd_cold |
| small | 1413 | 7828.8ms | 1413 | 394.0ms | 1.97x | 5.5ms | 98.5ms | 55.4ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
