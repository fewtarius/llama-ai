# Qwen3.6-27B-UD-Q8_K_XL (vulkan)

**Context:** 262144 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 62709.6ms | 15721 | 62730.3ms | 1.0x | 4.0ms | 4.0ms | 163.6ms | miss |
| medium | 5409 | 20324.1ms | 5409 | 20297.6ms | 1.0x | 3.8ms | 3.8ms | 162.4ms | miss |
| small | 1243 | 4623.6ms | 1243 | 4613.5ms | 1.0x | 3.7ms | 3.7ms | 167.2ms | miss |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
