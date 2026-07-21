# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15723 | 84474.2ms | 15723 | 467.4ms | 12.1x | 5.4ms | 116.8ms | 53.3ms | ssd_warm |
| medium | 5411 | 26539.6ms | 5411 | 291.8ms | 4.88x | 4.9ms | 72.9ms | 48.8ms | ssd_warm |
| small | 1244 | 6068.8ms | 1244 | 227.2ms | 1.9x | 4.9ms | 56.8ms | 46.8ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
