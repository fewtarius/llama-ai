# Benchmark Results: VULKAN

**Date:** 20260623-0647 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.48x (1851.6/261.9ms, ssd_cold) | TTFT 2.32x (6321.7/967.7ms, ssd_cold) | TTFT 4.38x (18748.6/1459.1ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 18748.6ms | 1459.1ms | 4.38x | 1.2ms | 364.8ms | 25.9ms | ssd_cold |
| medium | 6321.7ms | 967.7ms | 2.32x | 1.2ms | 241.9ms | 24.9ms | ssd_cold |
| small | 1851.6ms | 261.9ms | 1.48x | 1.5ms | 65.5ms | 25.8ms | ssd_cold |
