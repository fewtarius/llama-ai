# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 247728.6ms | 15721 | 10902.8ms | 11.05x | 15.8ms | 2725.7ms | 96.3ms | ssd_cold |
| medium | 5409 | 72569.5ms | 5409 | 995.5ms | 6.58x | 13.4ms | 248.9ms | 89.6ms | ssd_cold |
| small | 1243 | 15185.1ms | 1243 | 608.8ms | 2.22x | 12.2ms | 152.2ms | 85.8ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
