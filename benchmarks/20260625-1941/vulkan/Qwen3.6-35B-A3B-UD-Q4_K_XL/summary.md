# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 121541.1ms | 15721 | 988.9ms | 17.56x | 7.7ms | 247.2ms | 51.3ms | ssd_cold |
| medium | 5409 | 37854.9ms | 5409 | 613.7ms | 6.84x | 7.0ms | 153.4ms | 46.3ms | ssd_cold |
| small | 1243 | 8444.7ms | 1243 | 472.8ms | 2.34x | 6.8ms | 118.2ms | 44.0ms | ssd_cold |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
