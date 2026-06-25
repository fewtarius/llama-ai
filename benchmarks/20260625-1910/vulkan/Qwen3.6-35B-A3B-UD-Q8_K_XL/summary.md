# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 17503.9ms | 15721 | 1108.0ms | 4.43x | 1.1ms | 277.0ms | 29.4ms | ssd_cold |
| medium | 5409 | 6026.6ms | 5409 | 655.3ms | 2.29x | 1.1ms | 163.8ms | 26.4ms | ssd_cold |
| small | 1243 | 1681.9ms | 1243 | 337.4ms | 1.42x | 1.4ms | 84.4ms | 27.4ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
