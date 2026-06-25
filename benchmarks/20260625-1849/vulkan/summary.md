# Benchmark Results: VULKAN

**Date:** 20260625-1849 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 | TTFT 1.79x (2510.3/207.0ms, ssd_cold) | TTFT 3.29x (8967.2/615.7ms, ssd_cold) | TTFT 6.15x (27864.6/1455.2ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### gpt-oss-120b-UD-Q8_K_XL-00001-of-00002

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 27864.6ms | 1455.2ms | 6.15x | 1.8ms | 363.8ms | 28.5ms | ssd_cold |
| medium | 8967.2ms | 615.7ms | 3.29x | 1.7ms | 153.9ms | 25.1ms | ssd_cold |
| small | 2510.3ms | 207.0ms | 1.79x | 2.1ms | 51.7ms | 25.3ms | ssd_cold |
