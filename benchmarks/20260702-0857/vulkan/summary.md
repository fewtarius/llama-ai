# Benchmark Results: VULKAN

**Date:** 20260702-0857 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003 | TTFT 2.31x (6758.5/1189.0ms, ssd_warm) | TTFT 4.11x (30410.7/6069.5ms, ssd_warm) | TTFT 5.86x (107302.9/18395.4ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 107302.9ms | 18395.4ms | 5.86x | 7.1ms | 4598.8ms | 236.5ms | ssd_warm |
| medium | 30410.7ms | 6069.5ms | 4.11x | 5.9ms | 1517.4ms | 97.9ms | ssd_warm |
| small | 6758.5ms | 1189.0ms | 2.31x | 5.8ms | 297.2ms | 43.2ms | ssd_warm |
