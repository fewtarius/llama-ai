# Benchmark Results: VULKAN

**Date:** 20260622-1702 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.69x (2820.0/338.4ms, ssd_cold) | TTFT 2.86x (9484.3/847.0ms, ssd_cold) | TTFT 6.38x (27367.3/938.6ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 27367.3ms | 938.6ms | 6.38x | 1.7ms | 234.6ms | 26.3ms | ssd_cold |
| medium | 9484.3ms | 847.0ms | 2.86x | 1.8ms | 211.8ms | 25.2ms | ssd_cold |
| small | 2820.0ms | 338.4ms | 1.69x | 2.3ms | 84.6ms | 29.2ms | ssd_cold |
