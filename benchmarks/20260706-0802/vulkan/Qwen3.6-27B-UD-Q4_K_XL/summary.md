# Qwen3.6-27B-UD-Q4_K_XL (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 55834.5ms | 15721 | 311.5ms | 4.22x | 3.6ms | 77.9ms | 126.0ms | ssd_warm |
| medium | 5409 | 18851.6ms | 5409 | 205.5ms | 2.27x | 3.5ms | 51.4ms | 108.4ms | ssd_warm |
| small | 1243 | 5742.8ms | 1243 | 165.2ms | 1.39x | 4.6ms | 41.3ms | 101.8ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
