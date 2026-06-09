# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 270568.5ms | 15721 | 1382.2ms | 19.19x | 17.2ms | 345.5ms | 102.2ms | ssd_cold |
| medium | 5409 | 75723.0ms | 5409 | 812.3ms | 6.84x | 14.0ms | 203.1ms | 92.2ms | ssd_cold |
| small | 1243 | 15344.3ms | 1243 | 595.3ms | 2.22x | 12.3ms | 148.8ms | 86.7ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
