# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 20285.2ms | 15721 | 472.0ms | 6.73x | 1.3ms | 118.0ms | 49.2ms | ssd_warm |
| medium | 5409 | 6923.4ms | 5409 | 208.0ms | 3.17x | 1.3ms | 52.0ms | 34.0ms | ssd_warm |
| small | 1243 | 2254.5ms | 1243 | 140.5ms | 1.89x | 1.8ms | 35.1ms | 32.4ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
