# Benchmark Results: VULKAN

**Date:** 20260623-0636 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.46x (1880.2/328.3ms, ssd_cold) | TTFT 2.26x (6299.9/657.7ms, ssd_cold) | TTFT 4.61x (18261.4/1204.7ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 18261.4ms | 1204.7ms | 4.61x | 1.2ms | 301.2ms | 25.9ms | ssd_cold |
| medium | 6299.9ms | 657.7ms | 2.26x | 1.2ms | 164.4ms | 24.1ms | ssd_cold |
| small | 1880.2ms | 328.3ms | 1.46x | 1.5ms | 82.1ms | 26.8ms | ssd_cold |
