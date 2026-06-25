# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 21043.7ms | 17347 | 490.0ms | 6.69x | 1.2ms | 122.5ms | 27.5ms | ssd_cold |
| medium | 6083 | 6213.8ms | 6083 | 327.4ms | 2.83x | 1.0ms | 81.8ms | 25.3ms | ssd_cold |
| small | 1413 | 1915.2ms | 1413 | 255.9ms | 1.59x | 1.4ms | 64.0ms | 23.7ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
