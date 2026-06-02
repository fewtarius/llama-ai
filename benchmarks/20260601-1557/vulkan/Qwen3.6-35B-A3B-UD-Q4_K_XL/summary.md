# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 130060.1ms | 15721 | 1577.3ms | 15.65x | 8.3ms | 394.3ms | 52.0ms | ssd_cold |
| medium | 5409 | 40015.6ms | 5409 | 786.5ms | 6.4x | 7.4ms | 196.6ms | 48.4ms | ssd_cold |
| small | 1243 | 8864.7ms | 1243 | 495.6ms | 2.24x | 7.1ms | 123.9ms | 46.6ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
