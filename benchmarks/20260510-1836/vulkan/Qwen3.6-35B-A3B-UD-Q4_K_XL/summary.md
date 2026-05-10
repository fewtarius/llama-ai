# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 120392.6ms | 15721 | 1582.0ms | 15.69x | 7.7ms | 395.5ms | 46.8ms | ssd_cold |
| medium | 5409 | 38837.1ms | 5409 | 816.6ms | 6.82x | 7.2ms | 204.2ms | 43.5ms | ssd_cold |
| small | 1243 | 8734.3ms | 1243 | 522.3ms | 2.34x | 7.0ms | 130.6ms | 41.7ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
