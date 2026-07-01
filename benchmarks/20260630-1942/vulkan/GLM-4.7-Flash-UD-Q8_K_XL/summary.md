# GLM-4.7-Flash-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 40592.1ms | 15489 | 1072.6ms | 9.3x | 2.6ms | 268.1ms | 111.1ms | ssd_warm |
| medium | 5237 | 8197.1ms | 5237 | 320.5ms | 3.49x | 1.6ms | 80.1ms | 53.6ms | ssd_warm |
| small | 1145 | 1543.3ms | 1145 | 126.3ms | 1.59x | 1.3ms | 31.6ms | 34.1ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
