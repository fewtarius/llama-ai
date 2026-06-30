# Benchmark Results: VULKAN

**Date:** 20260630-1918 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003 | TTFT 2.59x (2886.2/155.5ms, ssd_warm) | TTFT 3.3x (9026.1/197.0ms, ssd_warm) | TTFT 9.2x (26308.5/306.5ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 26308.5ms | 306.5ms | 9.2x | 1.7ms | 76.6ms | 53.9ms | ssd_warm |
| medium | 9026.1ms | 197.0ms | 3.3x | 1.7ms | 49.2ms | 42.9ms | ssd_warm |
| small | 2886.2ms | 155.5ms | 2.59x | 2.3ms | 38.9ms | 46.1ms | ssd_warm |
