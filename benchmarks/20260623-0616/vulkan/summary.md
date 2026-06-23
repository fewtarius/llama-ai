# Benchmark Results: VULKAN

**Date:** 20260623-0616 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 | TTFT 1.78x (2737.1/200.8ms, ssd_cold) | TTFT 3.48x (9806.6/558.6ms, ssd_cold) | TTFT 6.27x (29917.5/1689.6ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### gpt-oss-120b-UD-Q8_K_XL-00001-of-00002

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 29917.5ms | 1689.6ms | 6.27x | 1.9ms | 422.4ms | 23.7ms | ssd_cold |
| medium | 9806.6ms | 558.6ms | 3.48x | 1.9ms | 139.6ms | 24.0ms | ssd_cold |
| small | 2737.1ms | 200.8ms | 1.78x | 2.3ms | 50.2ms | 24.0ms | ssd_cold |
