# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 143090.6ms | 15721 | 990.4ms | 18.91x | 9.1ms | 247.6ms | 53.6ms | ssd_cold |
| medium | 5409 | 43309.9ms | 5409 | 568.2ms | 7.14x | 8.0ms | 142.1ms | 48.8ms | ssd_cold |
| small | 1243 | 9318.4ms | 1243 | 405.3ms | 2.4x | 7.5ms | 101.3ms | 46.1ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
