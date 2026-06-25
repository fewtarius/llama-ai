# gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15380 | 27864.6ms | 15380 | 1455.2ms | 6.15x | 1.8ms | 363.8ms | 28.5ms | ssd_cold |
| medium | 5250 | 8967.2ms | 5250 | 615.7ms | 3.29x | 1.7ms | 153.9ms | 25.1ms | ssd_cold |
| small | 1205 | 2510.3ms | 1205 | 207.0ms | 1.79x | 2.1ms | 51.7ms | 25.3ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
