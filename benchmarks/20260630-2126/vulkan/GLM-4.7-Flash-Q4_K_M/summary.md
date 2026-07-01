# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15489 | 802116.1ms | 15489 | 1644.5ms | 20.08x | 51.8ms | 411.1ms | 467.8ms | ssd_warm |
| medium | 5237 | 112822.4ms | 5237 | 538.8ms | 8.8x | 21.5ms | 134.7ms | 234.7ms | ssd_warm |
| small | 1145 | 9021.8ms | 1145 | 244.5ms | 2.64x | 7.9ms | 61.1ms | 136.7ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
