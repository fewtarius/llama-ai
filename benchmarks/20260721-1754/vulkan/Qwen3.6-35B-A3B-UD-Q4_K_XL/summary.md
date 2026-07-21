# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15723 | 84019.5ms | 15723 | 469.1ms | 12.02x | 5.3ms | 117.3ms | 53.3ms | ssd_warm |
| medium | 5411 | 26680.1ms | 5411 | 292.9ms | 4.91x | 4.9ms | 73.2ms | 49.0ms | ssd_warm |
| small | 1244 | 6153.2ms | 1244 | 232.6ms | 1.91x | 4.9ms | 58.2ms | 47.3ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
