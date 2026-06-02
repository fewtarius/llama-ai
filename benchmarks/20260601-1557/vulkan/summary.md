# Benchmark Results: VULKAN

**Date:** 20260601-1557 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.24x (8864.7/495.6ms, ssd_cold) | TTFT 6.4x (40015.6/786.5ms, ssd_cold) | TTFT 15.65x (130060.1/1577.3ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 130060.1ms | 1577.3ms | 15.65x | 8.3ms | 394.3ms | 52.0ms | ssd_cold |
| medium | 40015.6ms | 786.5ms | 6.4x | 7.4ms | 196.6ms | 48.4ms | ssd_cold |
| small | 8864.7ms | 495.6ms | 2.24x | 7.1ms | 123.9ms | 46.6ms | ssd_cold |
