# MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15079 | 107302.9ms | 15079 | 18395.4ms | 5.86x | 7.1ms | 4598.8ms | 236.5ms | ssd_warm |
| medium | 5156 | 30410.7ms | 5156 | 6069.5ms | 4.11x | 5.9ms | 1517.4ms | 97.9ms | ssd_warm |
| small | 1170 | 6758.5ms | 1170 | 1189.0ms | 2.31x | 5.8ms | 297.2ms | 43.2ms | ssd_warm |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
