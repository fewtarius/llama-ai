# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 130858.9ms | 17347 | 1408.1ms | 13.09x | 7.5ms | 352.0ms | 72.4ms | ssd_cold |
| medium | 6083 | 38006.0ms | 6083 | 970.3ms | 4.91x | 6.2ms | 242.6ms | 65.5ms | ssd_cold |
| small | 1413 | 8530.9ms | 1413 | 708.7ms | 1.9x | 6.0ms | 177.2ms | 61.9ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
