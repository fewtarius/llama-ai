# gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15380 | 29917.5ms | 15380 | 1689.6ms | 6.27x | 1.9ms | 422.4ms | 23.7ms | ssd_cold |
| medium | 5250 | 9806.6ms | 5250 | 558.6ms | 3.48x | 1.9ms | 139.6ms | 24.0ms | ssd_cold |
| small | 1205 | 2737.1ms | 1205 | 200.8ms | 1.78x | 2.3ms | 50.2ms | 24.0ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
