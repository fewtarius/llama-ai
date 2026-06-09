# Benchmark Results: VULKAN

**Date:** 20260602-0230 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 3.3x (28206.8/586.7ms, ssd_cold) | TTFT 17.71x (357298.4/1870.7ms, ssd_cold) | TTFT 0x (0/93555.6ms, ssd_cold) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 2.13x (18895.1/1661.2ms, ssd_cold) | TTFT 7.3x (104071.5/1918.6ms, ssd_cold) | TTFT 28.61x (520443.9/3275.4ms, ssd_cold) |
| Qwen3.5-27B-UD-Q5_K_XL | TTFT 1.84x (54619.8/1999.9ms, ssd_cold) | TTFT 4.59x (241283.2/3997.9ms, ssd_cold) | TTFT 11.88x (779802.3/6005.6ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.22x (15185.1/608.8ms, ssd_cold) | TTFT 6.58x (72569.5/995.5ms, ssd_cold) | TTFT 11.05x (247728.6/10902.8ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 0ms | 93555.6ms | 0x | 0ms | 359.8ms | 0ms | ssd_cold |
| medium | 357298.4ms | 1870.7ms | 17.71x | 68.2ms | 467.7ms | 149.9ms | ssd_cold |
| small | 28206.8ms | 586.7ms | 3.3x | 24.6ms | 146.7ms | 89.5ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 520443.9ms | 3275.4ms | 28.61x | 30.0ms | 818.9ms | 117.8ms | ssd_cold |
| medium | 104071.5ms | 1918.6ms | 7.3x | 17.1ms | 479.7ms | 109.4ms | ssd_cold |
| small | 18895.1ms | 1661.2ms | 2.13x | 13.4ms | 415.3ms | 105.5ms | ssd_cold |

### Qwen3.5-27B-UD-Q5_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 779802.3ms | 6005.6ms | 11.88x | 49.6ms | 1501.4ms | 505.3ms | ssd_cold |
| medium | 241283.2ms | 3997.9ms | 4.59x | 44.6ms | 999.5ms | 483.0ms | ssd_cold |
| small | 54619.8ms | 1999.9ms | 1.84x | 43.9ms | 500.0ms | 475.9ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 247728.6ms | 10902.8ms | 11.05x | 15.8ms | 2725.7ms | 96.3ms | ssd_cold |
| medium | 72569.5ms | 995.5ms | 6.58x | 13.4ms | 248.9ms | 89.6ms | ssd_cold |
| small | 15185.1ms | 608.8ms | 2.22x | 12.2ms | 152.2ms | 85.8ms | ssd_cold |
