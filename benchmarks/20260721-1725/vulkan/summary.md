# Benchmark Results: VULKAN

**Date:** 20260721-1725 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 1.9x (6068.8/227.2ms, ssd_warm) | TTFT 4.88x (26539.6/291.8ms, ssd_warm) | TTFT 12.1x (84474.2/467.4ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 84474.2ms | 467.4ms | 12.1x | 5.4ms | 116.8ms | 53.3ms | ssd_warm |
| medium | 26539.6ms | 291.8ms | 4.88x | 4.9ms | 72.9ms | 48.8ms | ssd_warm |
| small | 6068.8ms | 227.2ms | 1.9x | 4.9ms | 56.8ms | 46.8ms | ssd_warm |
