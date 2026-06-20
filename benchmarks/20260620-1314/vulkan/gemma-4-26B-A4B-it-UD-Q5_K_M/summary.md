# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 29676.2ms | 17347 | 28828.1ms | 1.03x | 1.7ms | 1.7ms | 25.6ms | miss |
| medium | 6083 | 8937.5ms | 6083 | 8570.6ms | 1.03x | 1.5ms | 1.4ms | 23.7ms | miss |
| small | 1413 | 5794.2ms | 1413 | 1991.5ms | 1.78x | 4.1ms | 1.4ms | 22.9ms | miss |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
