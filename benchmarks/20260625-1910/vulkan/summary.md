# Benchmark Results: VULKAN

**Date:** 20260625-1910 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.42x (1681.9/337.4ms, ssd_cold) | TTFT 2.29x (6026.6/655.3ms, ssd_cold) | TTFT 4.43x (17503.9/1108.0ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 17503.9ms | 1108.0ms | 4.43x | 1.1ms | 277.0ms | 29.4ms | ssd_cold |
| medium | 6026.6ms | 655.3ms | 2.29x | 1.1ms | 163.8ms | 26.4ms | ssd_cold |
| small | 1681.9ms | 337.4ms | 1.42x | 1.4ms | 84.4ms | 27.4ms | ssd_cold |
