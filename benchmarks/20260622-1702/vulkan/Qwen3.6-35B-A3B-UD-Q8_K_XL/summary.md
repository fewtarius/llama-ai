# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 27367.3ms | 15721 | 938.6ms | 6.38x | 1.7ms | 234.6ms | 26.3ms | ssd_cold |
| medium | 5409 | 9484.3ms | 5409 | 847.0ms | 2.86x | 1.8ms | 211.8ms | 25.2ms | ssd_cold |
| small | 1243 | 2820.0ms | 1243 | 338.4ms | 1.69x | 2.3ms | 84.6ms | 29.2ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
