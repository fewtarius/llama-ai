# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 22859.5ms | 17347 | 123.0ms | 8.78x | 1.3ms | 30.7ms | 42.1ms | ssd_warm |
| medium | 6083 | 7669.6ms | 6083 | 83.1ms | 4.02x | 1.3ms | 20.8ms | 35.0ms | ssd_warm |
| small | 1413 | 1514.8ms | 1413 | 70.7ms | 1.97x | 1.1ms | 17.7ms | 32.6ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
