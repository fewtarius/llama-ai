# GLM-4.7-Flash-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 41124.1ms | 15489 | 2342.2ms | 6.34x | 2.7ms | 585.6ms | 34.3ms | ssd_cold |
| medium | 5237 | 9097.4ms | 5237 | 732.6ms | 2.76x | 1.7ms | 183.1ms | 31.4ms | ssd_cold |
| small | 1145 | 1850.4ms | 1145 | 242.6ms | 1.37x | 1.6ms | 60.6ms | 28.2ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
