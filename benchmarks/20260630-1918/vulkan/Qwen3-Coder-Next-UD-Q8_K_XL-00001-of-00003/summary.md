# Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15663 | 26308.5ms | 15663 | 306.5ms | 9.2x | 1.7ms | 76.6ms | 53.9ms | ssd_warm |
| medium | 5398 | 9026.1ms | 5398 | 197.0ms | 3.3x | 1.7ms | 49.2ms | 42.9ms | ssd_warm |
| small | 1251 | 2886.2ms | 1251 | 155.5ms | 2.59x | 2.3ms | 38.9ms | 46.1ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
