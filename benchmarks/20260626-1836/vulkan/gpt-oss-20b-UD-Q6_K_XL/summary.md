# gpt-oss-20b-UD-Q6_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15380 | 12368.8ms | 15380 | 156.3ms | 7.83x | 0.8ms | 39.1ms | 32.2ms | ssd_warm |
| medium | 5250 | 3664.7ms | 5250 | 86.4ms | 3.38x | 0.7ms | 21.6ms | 19.8ms | ssd_warm |
| small | 1205 | 1111.2ms | 1205 | 38.5ms | 2.01x | 0.9ms | 9.6ms | 17.3ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
