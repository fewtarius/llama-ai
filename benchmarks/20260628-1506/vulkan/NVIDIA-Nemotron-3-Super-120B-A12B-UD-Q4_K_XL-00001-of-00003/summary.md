# NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17810 | 71552.0ms | 17810 | 492.9ms | 15.41x | 4.0ms | 123.2ms | 118.1ms | ssd_warm |
| medium | 6252 | 25332.9ms | 6252 | 360.7ms | 5.39x | 4.1ms | 90.2ms | 85.4ms | ssd_warm |
| small | 1457 | 7709.1ms | 1457 | 320.4ms | 2.78x | 5.3ms | 80.1ms | 96.2ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
