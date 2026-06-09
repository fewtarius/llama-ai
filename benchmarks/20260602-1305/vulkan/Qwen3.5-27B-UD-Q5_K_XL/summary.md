# Qwen3.5-27B-UD-Q5_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 875971.3ms | 15721 | 4114.1ms | 13.34x | 55.7ms | 1028.5ms | 518.3ms | ssd_cold |
| medium | 5409 | 250271.8ms | 5409 | 2530.9ms | 4.77x | 46.3ms | 632.7ms | 489.8ms | ssd_cold |
| small | 1243 | 55567.0ms | 1243 | 1857.6ms | 1.86x | 44.7ms | 464.4ms | 480.6ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
