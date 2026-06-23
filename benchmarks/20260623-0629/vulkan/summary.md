# Benchmark Results: VULKAN

**Date:** 20260623-0629 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-UD-Q8_K_XL | TTFT 1.37x (1850.4/242.6ms, ssd_cold) | TTFT 2.76x (9097.4/732.6ms, ssd_cold) | TTFT 6.34x (41124.1/2342.2ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 41124.1ms | 2342.2ms | 6.34x | 2.7ms | 585.6ms | 34.3ms | ssd_cold |
| medium | 9097.4ms | 732.6ms | 2.76x | 1.7ms | 183.1ms | 31.4ms | ssd_cold |
| small | 1850.4ms | 242.6ms | 1.37x | 1.6ms | 60.6ms | 28.2ms | ssd_cold |
