# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 315506.0ms | 17347 | 1869.7ms | 18.8x | 18.2ms | 467.4ms | 121.1ms | ssd_cold |
| medium | 6083 | 77459.2ms | 6083 | 1459.2ms | 5.75x | 12.7ms | 364.8ms | 113.2ms | ssd_cold |
| small | 1413 | 15236.6ms | 1413 | 1016.8ms | 1.95x | 10.8ms | 254.2ms | 107.3ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
