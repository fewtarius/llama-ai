# Benchmark Results: VULKAN

**Date:** 20260611-0656 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 2.42x (9737.0/343.5ms, ssd_cold) | TTFT 7.24x (74198.4/1020.0ms, ssd_cold) | TTFT 19.95x (467588.2/2686.1ms, ssd_cold) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.9x (8530.9/708.7ms, ssd_cold) | TTFT 4.91x (38006.0/970.3ms, ssd_cold) | TTFT 13.09x (130858.9/1408.1ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.4x (9318.4/405.3ms, ssd_cold) | TTFT 7.14x (43309.9/568.2ms, ssd_cold) | TTFT 18.91x (143090.6/990.4ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 467588.2ms | 2686.1ms | 19.95x | 30.2ms | 671.5ms | 175.0ms | ssd_cold |
| medium | 74198.4ms | 1020.0ms | 7.24x | 14.2ms | 255.0ms | 82.8ms | ssd_cold |
| small | 9737.0ms | 343.5ms | 2.42x | 8.5ms | 85.9ms | 49.5ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 130858.9ms | 1408.1ms | 13.09x | 7.5ms | 352.0ms | 72.4ms | ssd_cold |
| medium | 38006.0ms | 970.3ms | 4.91x | 6.2ms | 242.6ms | 65.5ms | ssd_cold |
| small | 8530.9ms | 708.7ms | 1.9x | 6.0ms | 177.2ms | 61.9ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 143090.6ms | 990.4ms | 18.91x | 9.1ms | 247.6ms | 53.6ms | ssd_cold |
| medium | 43309.9ms | 568.2ms | 7.14x | 8.0ms | 142.1ms | 48.8ms | ssd_cold |
| small | 9318.4ms | 405.3ms | 2.4x | 7.5ms | 101.3ms | 46.1ms | ssd_cold |
