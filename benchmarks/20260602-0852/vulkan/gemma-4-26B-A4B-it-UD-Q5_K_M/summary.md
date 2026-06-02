# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 114809.5ms | 17347 | 1451.1ms | 11.5x | 6.6ms | 362.8ms | 69.0ms | ssd_cold |
| medium | 6083 | 34285.9ms | 6083 | 1044.7ms | 4.54x | 5.6ms | 261.2ms | 63.3ms | ssd_cold |
| small | 1413 | 8128.8ms | 1413 | 761.6ms | 1.84x | 5.8ms | 190.4ms | 60.5ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
