# Qwen3.6-27B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15723 | 271495.1ms | 15723 | 1027.3ms | 9.89x | 17.3ms | 256.8ms | 227.0ms | ssd_warm |
| medium | 5411 | 87783.4ms | 5411 | 669.8ms | 4.13x | 16.2ms | 167.5ms | 211.6ms | ssd_warm |
| small | 1244 | 19910.5ms | 1244 | 598.0ms | 1.71x | 16.0ms | 149.5ms | 203.6ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
