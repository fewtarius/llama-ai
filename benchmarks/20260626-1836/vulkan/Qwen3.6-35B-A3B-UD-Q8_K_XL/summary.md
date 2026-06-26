# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 19659.6ms | 15721 | 561.1ms | 6.13x | 1.3ms | 140.3ms | 44.8ms | ssd_warm |
| medium | 5409 | 6898.9ms | 5409 | 249.7ms | 3.08x | 1.3ms | 62.4ms | 34.8ms | ssd_warm |
| small | 1243 | 2199.8ms | 1243 | 155.3ms | 1.91x | 1.8ms | 38.8ms | 32.4ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
