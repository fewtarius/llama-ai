# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17349 | 253195.9ms | 17349 | 652.8ms | 14.5x | 14.6ms | 163.2ms | 303.9ms | ssd_warm |
| medium | 6085 | 65638.7ms | 6085 | 367.3ms | 6.31x | 10.8ms | 91.8ms | 238.9ms | ssd_warm |
| small | 1414 | 7226.5ms | 1414 | 315.4ms | 2.08x | 5.1ms | 78.8ms | 163.8ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
