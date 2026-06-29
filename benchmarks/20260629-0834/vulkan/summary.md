# Benchmark Results: VULKAN

**Date:** 20260629-0834 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.92x (2239.7/136.6ms, ssd_warm) | TTFT 3.1x (6945.4/215.5ms, ssd_warm) | TTFT 6.0x (19563.4/525.2ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 19563.4ms | 525.2ms | 6.0x | 1.2ms | 131.3ms | 45.3ms | ssd_warm |
| medium | 6945.4ms | 215.5ms | 3.1x | 1.3ms | 53.9ms | 34.1ms | ssd_warm |
| small | 2239.7ms | 136.6ms | 1.92x | 1.8ms | 34.1ms | 32.6ms | ssd_warm |
