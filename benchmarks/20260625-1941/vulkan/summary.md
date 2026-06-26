# Benchmark Results: VULKAN

**Date:** 20260625-1941 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 2.64x (10285.0/350.1ms, ssd_cold) | TTFT 7.87x (74375.3/1022.5ms, ssd_cold) | TTFT 20.49x (439068.3/2680.1ms, ssd_cold) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.97x (7828.8/394.0ms, ssd_cold) | TTFT 5.32x (33980.2/514.5ms, ssd_cold) | TTFT 13.5x (113358.9/806.8ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.34x (8444.7/472.8ms, ssd_cold) | TTFT 6.84x (37854.9/613.7ms, ssd_cold) | TTFT 17.56x (121541.1/988.9ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 439068.3ms | 2680.1ms | 20.49x | 28.3ms | 670.0ms | 161.9ms | ssd_cold |
| medium | 74375.3ms | 1022.5ms | 7.87x | 14.2ms | 255.6ms | 77.7ms | ssd_cold |
| small | 10285.0ms | 350.1ms | 2.64x | 9.0ms | 87.5ms | 44.5ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 113358.9ms | 806.8ms | 13.5x | 6.5ms | 201.7ms | 62.2ms | ssd_cold |
| medium | 33980.2ms | 514.5ms | 5.32x | 5.6ms | 128.6ms | 57.5ms | ssd_cold |
| small | 7828.8ms | 394.0ms | 1.97x | 5.5ms | 98.5ms | 55.4ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 121541.1ms | 988.9ms | 17.56x | 7.7ms | 247.2ms | 51.3ms | ssd_cold |
| medium | 37854.9ms | 613.7ms | 6.84x | 7.0ms | 153.4ms | 46.3ms | ssd_cold |
| small | 8444.7ms | 472.8ms | 2.34x | 6.8ms | 118.2ms | 44.0ms | ssd_cold |
