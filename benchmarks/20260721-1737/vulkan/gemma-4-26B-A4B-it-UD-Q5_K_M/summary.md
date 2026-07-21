# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17349 | 94784.8ms | 17349 | 535.4ms | 11.5x | 5.5ms | 133.8ms | 163.3ms | ssd_warm |
| medium | 6085 | 30698.8ms | 6085 | 362.1ms | 5.66x | 5.0ms | 90.5ms | 159.3ms | ssd_warm |
| small | 1414 | 7380.2ms | 1414 | 271.2ms | 3.19x | 5.2ms | 67.8ms | 153.6ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
