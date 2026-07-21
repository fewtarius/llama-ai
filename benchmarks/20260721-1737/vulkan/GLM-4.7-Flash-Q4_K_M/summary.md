# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 0 | 0ms | 0 | 0ms | 0x | 0ms | 0ms | 0ms | miss |
| medium | 0 | 0ms | 0 | 0ms | 0x | 0ms | 0ms | 0ms | miss |
| small | 0 | 0ms | 0 | 0ms | 0x | 0ms | 0ms | 0ms | miss |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
