# Qwen3.6-27B-UD-Q8_K_XL (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 63696.1ms | 15721 | 2198.5ms | 3.37x | 4.1ms | 549.6ms | 167.2ms | ssd_cold |
| medium | 5409 | 21052.4ms | 5409 | 1242.4ms | 1.95x | 3.9ms | 310.6ms | 163.8ms | ssd_cold |
| small | 1243 | 5360.4ms | 1243 | 850.6ms | 1.15x | 4.3ms | 212.7ms | 165.2ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
