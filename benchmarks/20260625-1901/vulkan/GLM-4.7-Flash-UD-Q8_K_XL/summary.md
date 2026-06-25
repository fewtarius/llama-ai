# GLM-4.7-Flash-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 41970.7ms | 15489 | 2667.3ms | 6.68x | 2.7ms | 666.8ms | 46.4ms | ssd_cold |
| medium | 5237 | 8351.4ms | 5237 | 828.4ms | 2.59x | 1.6ms | 207.1ms | 32.8ms | ssd_cold |
| small | 1145 | 1700.2ms | 1145 | 236.3ms | 1.47x | 1.5ms | 59.1ms | 30.7ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
