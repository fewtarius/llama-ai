# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17347 | 123352.1ms | 17347 | 2485.2ms | 10.81x | 7.1ms | 621.3ms | 73.2ms | ssd_cold |
| medium | 6083 | 40612.7ms | 6083 | 1635.2ms | 5.06x | 6.7ms | 408.8ms | 60.5ms | ssd_cold |
| small | 1413 | 10180.2ms | 1413 | 1334.2ms | 2.03x | 7.2ms | 333.5ms | 55.0ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
