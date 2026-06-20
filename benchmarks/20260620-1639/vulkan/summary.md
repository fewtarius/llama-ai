# Benchmark Results: VULKAN

**Date:** 20260620-1639 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 1.84x (2083.1/154.9ms, ssd_cold) | TTFT 4.71x (11812.2/531.6ms, ssd_cold) | TTFT 12.61x (59602.6/1590.0ms, ssd_cold) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.6x (2205.6/304.4ms, ssd_cold) | TTFT 3.58x (9285.8/321.9ms, ssd_cold) | TTFT 8.7x (31008.9/539.5ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 1.68x (1974.8/214.7ms, ssd_cold) | TTFT 3.47x (7720.9/415.8ms, ssd_cold) | TTFT 7.8x (23632.1/745.3ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 59602.6ms | 1590.0ms | 12.61x | 3.8ms | 397.5ms | 26.5ms | ssd_cold |
| medium | 11812.2ms | 531.6ms | 4.71x | 2.3ms | 132.9ms | 19.5ms | ssd_cold |
| small | 2083.1ms | 154.9ms | 1.84x | 1.8ms | 38.7ms | 16.7ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 31008.9ms | 539.5ms | 8.7x | 1.8ms | 134.9ms | 26.5ms | ssd_cold |
| medium | 9285.8ms | 321.9ms | 3.58x | 1.5ms | 80.5ms | 24.2ms | ssd_cold |
| small | 2205.6ms | 304.4ms | 1.6x | 1.6ms | 76.1ms | 23.8ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 23632.1ms | 745.3ms | 7.8x | 1.5ms | 186.3ms | 20.4ms | ssd_cold |
| medium | 7720.9ms | 415.8ms | 3.47x | 1.4ms | 104.0ms | 19.1ms | ssd_cold |
| small | 1974.8ms | 214.7ms | 1.68x | 1.6ms | 53.7ms | 18.9ms | ssd_cold |
