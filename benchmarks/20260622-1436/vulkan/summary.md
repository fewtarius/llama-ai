# Benchmark Results: VULKAN

**Date:** 20260622-1436 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 | TTFT 2.37x (4771.6/144.7ms, ssd_cold) | TTFT 4.72x (15823.5/342.2ms, ssd_cold) | TTFT 10.92x (48367.2/858.3ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### gpt-oss-120b-UD-Q8_K_XL-00001-of-00002

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 48367.2ms | 858.3ms | 10.92x | 3.1ms | 214.6ms | 26.1ms | ssd_cold |
| medium | 15823.5ms | 342.2ms | 4.72x | 3.0ms | 85.6ms | 24.6ms | ssd_cold |
| small | 4771.6ms | 144.7ms | 2.37x | 4.0ms | 36.2ms | 27.0ms | ssd_cold |
