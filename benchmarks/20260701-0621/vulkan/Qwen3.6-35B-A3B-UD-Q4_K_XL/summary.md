# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15723 | 199081.1ms | 15723 | 486.5ms | 16.48x | 12.7ms | 121.6ms | 194.1ms | ssd_warm |
| medium | 5411 | 31088.0ms | 5411 | 267.6ms | 3.73x | 5.7ms | 66.9ms | 163.7ms | ssd_warm |
| small | 1244 | 6249.4ms | 1244 | 226.6ms | 2.03x | 5.0ms | 56.6ms | 44.4ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
