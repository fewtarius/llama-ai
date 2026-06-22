# Benchmark Results: VULKAN

**Date:** 20260622-1527 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-UD-Q8_K_XL | TTFT 1.69x (3020.0/250.5ms, ssd_cold) | TTFT 3.21x (13208.4/1026.8ms, ssd_cold) | TTFT 8.73x (65165.4/2367.8ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 65165.4ms | 2367.8ms | 8.73x | 4.2ms | 591.9ms | 38.0ms | ssd_cold |
| medium | 13208.4ms | 1026.8ms | 3.21x | 2.5ms | 256.7ms | 32.3ms | ssd_cold |
| small | 3020.0ms | 250.5ms | 1.69x | 2.6ms | 62.6ms | 32.0ms | ssd_cold |
