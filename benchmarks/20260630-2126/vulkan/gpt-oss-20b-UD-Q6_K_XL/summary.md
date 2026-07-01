# gpt-oss-20b-UD-Q6_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15381 | 169487.7ms | 15381 | 729.5ms | 8.8x | 11.0ms | 182.4ms | 187.1ms | ssd_warm |
| medium | 5251 | 45417.0ms | 5251 | 429.2ms | 3.65x | 8.6ms | 107.3ms | 155.7ms | ssd_warm |
| small | 1205 | 6653.2ms | 1205 | 131.2ms | 1.39x | 5.5ms | 32.8ms | 126.6ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
