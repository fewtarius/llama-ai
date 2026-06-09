# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 0 | 0ms | 0 | 0ms | 0x | 0ms | 0ms | 0ms | miss |
| medium | 5237 | 607780.2ms | 5237 | 1605.0ms | 20.05x | 116.1ms | 401.2ms | 228.1ms | ssd_cold |
| small | 1145 | 34330.6ms | 1145 | 536.8ms | 3.32x | 30.0ms | 134.2ms | 108.2ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
