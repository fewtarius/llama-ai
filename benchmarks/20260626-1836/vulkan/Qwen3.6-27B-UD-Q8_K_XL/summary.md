# Qwen3.6-27B-UD-Q8_K_XL (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 66590.4ms | 15721 | 1140.5ms | 3.87x | 4.2ms | 285.1ms | 209.3ms | ssd_warm |
| medium | 5409 | 22806.5ms | 5409 | 501.4ms | 1.99x | 4.2ms | 125.3ms | 183.1ms | ssd_warm |
| small | 1243 | 6596.1ms | 1243 | 418.6ms | 1.29x | 5.3ms | 104.6ms | 183.2ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
