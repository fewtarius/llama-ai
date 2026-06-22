# gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15380 | 48367.2ms | 15380 | 858.3ms | 10.92x | 3.1ms | 214.6ms | 26.1ms | ssd_cold |
| medium | 5250 | 15823.5ms | 5250 | 342.2ms | 4.72x | 3.0ms | 85.6ms | 24.6ms | ssd_cold |
| small | 1205 | 4771.6ms | 1205 | 144.7ms | 2.37x | 4.0ms | 36.2ms | 27.0ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
