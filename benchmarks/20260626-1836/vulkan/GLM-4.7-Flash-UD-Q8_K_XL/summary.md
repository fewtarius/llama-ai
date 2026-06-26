# GLM-4.7-Flash-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 38963.3ms | 15489 | 1082.8ms | 9.49x | 2.5ms | 270.7ms | 103.8ms | ssd_warm |
| medium | 5237 | 7974.3ms | 5237 | 384.4ms | 3.22x | 1.5ms | 96.1ms | 52.8ms | ssd_warm |
| small | 1145 | 1513.3ms | 1145 | 177.5ms | 1.49x | 1.3ms | 44.4ms | 33.3ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
