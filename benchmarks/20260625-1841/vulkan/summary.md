# Benchmark Results: VULKAN

**Date:** 20260625-1841 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.59x (1915.2/255.9ms, ssd_cold) | TTFT 2.83x (6213.8/327.4ms, ssd_cold) | TTFT 6.69x (21043.7/490.0ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 21043.7ms | 490.0ms | 6.69x | 1.2ms | 122.5ms | 27.5ms | ssd_cold |
| medium | 6213.8ms | 327.4ms | 2.83x | 1.0ms | 81.8ms | 25.3ms | ssd_cold |
| small | 1915.2ms | 255.9ms | 1.59x | 1.4ms | 64.0ms | 23.7ms | ssd_cold |
