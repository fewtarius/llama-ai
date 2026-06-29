# Benchmark Results: VULKAN

**Date:** 20260628-1506 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-00001-of-00003 | TTFT 2.78x (7709.1/320.4ms, ssd_warm) | TTFT 5.39x (25332.9/360.7ms, ssd_warm) | TTFT 15.41x (71552.0/492.9ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-00001-of-00003

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 71552.0ms | 492.9ms | 15.41x | 4.0ms | 123.2ms | 118.1ms | ssd_warm |
| medium | 25332.9ms | 360.7ms | 5.39x | 4.1ms | 90.2ms | 85.4ms | ssd_warm |
| small | 7709.1ms | 320.4ms | 2.78x | 5.3ms | 80.1ms | 96.2ms | ssd_warm |
