# Benchmark Results: VULKAN

**Date:** 20260625-1901 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-UD-Q8_K_XL | TTFT 1.47x (1700.2/236.3ms, ssd_cold) | TTFT 2.59x (8351.4/828.4ms, ssd_cold) | TTFT 6.68x (41970.7/2667.3ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 41970.7ms | 2667.3ms | 6.68x | 2.7ms | 666.8ms | 46.4ms | ssd_cold |
| medium | 8351.4ms | 828.4ms | 2.59x | 1.6ms | 207.1ms | 32.8ms | ssd_cold |
| small | 1700.2ms | 236.3ms | 1.47x | 1.5ms | 59.1ms | 30.7ms | ssd_cold |
