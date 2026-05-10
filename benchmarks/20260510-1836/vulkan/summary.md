# Benchmark Results: VULKAN

**Date:** 20260510-1836 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 2.7x (9743.5/462.9ms, ssd_cold) | TTFT 7.39x (55883.9/1754.3ms, ssd_cold) | TTFT 17.97x (254908.1/4130.8ms, ssd_cold) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 2.03x (10180.2/1334.2ms, ssd_cold) | TTFT 5.06x (40612.7/1635.2ms, ssd_cold) | TTFT 10.81x (123352.1/2485.2ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.34x (8734.3/522.3ms, ssd_cold) | TTFT 6.82x (38837.1/816.6ms, ssd_cold) | TTFT 15.69x (120392.6/1582.0ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 254908.1ms | 4130.8ms | 17.97x | 16.5ms | 1032.7ms | 81.8ms | ssd_cold |
| medium | 55883.9ms | 1754.3ms | 7.39x | 10.7ms | 438.6ms | 51.1ms | ssd_cold |
| small | 9743.5ms | 462.9ms | 2.7x | 8.5ms | 115.7ms | 38.3ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 123352.1ms | 2485.2ms | 10.81x | 7.1ms | 621.3ms | 73.2ms | ssd_cold |
| medium | 40612.7ms | 1635.2ms | 5.06x | 6.7ms | 408.8ms | 60.5ms | ssd_cold |
| small | 10180.2ms | 1334.2ms | 2.03x | 7.2ms | 333.5ms | 55.0ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 120392.6ms | 1582.0ms | 15.69x | 7.7ms | 395.5ms | 46.8ms | ssd_cold |
| medium | 38837.1ms | 816.6ms | 6.82x | 7.2ms | 204.2ms | 43.5ms | ssd_cold |
| small | 8734.3ms | 522.3ms | 2.34x | 7.0ms | 130.6ms | 41.7ms | ssd_cold |
