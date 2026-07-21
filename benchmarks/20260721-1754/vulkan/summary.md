# Benchmark Results: VULKAN

**Date:** 20260721-1754 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 1.91x (6153.2/232.6ms, ssd_warm) | TTFT 4.91x (26680.1/292.9ms, ssd_warm) | TTFT 12.02x (84019.5/469.1ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 84019.5ms | 469.1ms | 12.02x | 5.3ms | 117.3ms | 53.3ms | ssd_warm |
| medium | 26680.1ms | 292.9ms | 4.91x | 4.9ms | 73.2ms | 49.0ms | ssd_warm |
| small | 6153.2ms | 232.6ms | 1.91x | 4.9ms | 58.2ms | 47.3ms | ssd_warm |
