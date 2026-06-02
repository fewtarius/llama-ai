# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 419528.2ms | 15489 | 2820.0ms | 18.71x | 27.1ms | 705.0ms | 159.2ms | ssd_cold |
| medium | 5237 | 66595.3ms | 5237 | 1122.0ms | 6.63x | 12.7ms | 280.5ms | 79.8ms | ssd_cold |
| small | 1145 | 9067.7ms | 1145 | 408.6ms | 2.28x | 7.9ms | 102.2ms | 49.1ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
