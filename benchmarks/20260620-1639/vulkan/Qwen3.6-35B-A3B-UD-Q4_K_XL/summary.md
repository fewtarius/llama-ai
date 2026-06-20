# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 23632.1ms | 15721 | 745.3ms | 7.8x | 1.5ms | 186.3ms | 20.4ms | ssd_cold |
| medium | 5409 | 7720.9ms | 5409 | 415.8ms | 3.47x | 1.4ms | 104.0ms | 19.1ms | ssd_cold |
| small | 1243 | 1974.8ms | 1243 | 214.7ms | 1.68x | 1.6ms | 53.7ms | 18.9ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
