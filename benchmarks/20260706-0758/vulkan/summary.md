# Benchmark Results: VULKAN

**Date:** 20260706-0758 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-27B-UD-Q4_K_XL | TTFT 1.71x (19910.5/598.0ms, ssd_warm) | TTFT 4.13x (87783.4/669.8ms, ssd_warm) | TTFT 9.89x (271495.1/1027.3ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-27B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 271495.1ms | 1027.3ms | 9.89x | 17.3ms | 256.8ms | 227.0ms | ssd_warm |
| medium | 87783.4ms | 669.8ms | 4.13x | 16.2ms | 167.5ms | 211.6ms | ssd_warm |
| small | 19910.5ms | 598.0ms | 1.71x | 16.0ms | 149.5ms | 203.6ms | ssd_warm |
