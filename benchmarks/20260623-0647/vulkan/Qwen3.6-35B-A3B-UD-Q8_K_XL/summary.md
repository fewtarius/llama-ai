# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 18748.6ms | 15721 | 1459.1ms | 4.38x | 1.2ms | 364.8ms | 25.9ms | ssd_cold |
| medium | 5409 | 6321.7ms | 5409 | 967.7ms | 2.32x | 1.2ms | 241.9ms | 24.9ms | ssd_cold |
| small | 1243 | 1851.6ms | 1243 | 261.9ms | 1.48x | 1.5ms | 65.5ms | 25.8ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
