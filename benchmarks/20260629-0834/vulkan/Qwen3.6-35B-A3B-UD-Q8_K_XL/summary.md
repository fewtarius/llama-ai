# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 19563.4ms | 15721 | 525.2ms | 6.0x | 1.2ms | 131.3ms | 45.3ms | ssd_warm |
| medium | 5409 | 6945.4ms | 5409 | 215.5ms | 3.1x | 1.3ms | 53.9ms | 34.1ms | ssd_warm |
| small | 1243 | 2239.7ms | 1243 | 136.6ms | 1.92x | 1.8ms | 34.1ms | 32.6ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
