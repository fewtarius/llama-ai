# Benchmark Results: VULKAN

**Date:** 20260701-0621 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.03x (6249.4/226.6ms, ssd_warm) | TTFT 3.73x (31088.0/267.6ms, ssd_warm) | TTFT 16.48x (199081.1/486.5ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 199081.1ms | 486.5ms | 16.48x | 12.7ms | 121.6ms | 194.1ms | ssd_warm |
| medium | 31088.0ms | 267.6ms | 3.73x | 5.7ms | 66.9ms | 163.7ms | ssd_warm |
| small | 6249.4ms | 226.6ms | 2.03x | 5.0ms | 56.6ms | 44.4ms | ssd_warm |
