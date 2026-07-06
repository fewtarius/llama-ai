# Benchmark Results: VULKAN

**Date:** 20260706-0802 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-27B-UD-Q4_K_XL | TTFT 1.39x (5742.8/165.2ms, ssd_warm) | TTFT 2.27x (18851.6/205.5ms, ssd_warm) | TTFT 4.22x (55834.5/311.5ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-27B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 55834.5ms | 311.5ms | 4.22x | 3.6ms | 77.9ms | 126.0ms | ssd_warm |
| medium | 18851.6ms | 205.5ms | 2.27x | 3.5ms | 51.4ms | 108.4ms | ssd_warm |
| small | 5742.8ms | 165.2ms | 1.39x | 4.6ms | 41.3ms | 101.8ms | ssd_warm |
