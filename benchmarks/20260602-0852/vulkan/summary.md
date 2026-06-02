# Benchmark Results: VULKAN

**Date:** 20260602-0852 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 2.28x (9067.7/408.6ms, ssd_cold) | TTFT 6.63x (66595.3/1122.0ms, ssd_cold) | TTFT 18.71x (419528.2/2820.0ms, ssd_cold) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.84x (8128.8/761.6ms, ssd_cold) | TTFT 4.54x (34285.9/1044.7ms, ssd_cold) | TTFT 11.5x (114809.5/1451.1ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.25x (8762.5/436.0ms, ssd_cold) | TTFT 6.35x (39052.2/630.7ms, ssd_cold) | TTFT 15.85x (125117.4/1061.7ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 419528.2ms | 2820.0ms | 18.71x | 27.1ms | 705.0ms | 159.2ms | ssd_cold |
| medium | 66595.3ms | 1122.0ms | 6.63x | 12.7ms | 280.5ms | 79.8ms | ssd_cold |
| small | 9067.7ms | 408.6ms | 2.28x | 7.9ms | 102.2ms | 49.1ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 114809.5ms | 1451.1ms | 11.5x | 6.6ms | 362.8ms | 69.0ms | ssd_cold |
| medium | 34285.9ms | 1044.7ms | 4.54x | 5.6ms | 261.2ms | 63.3ms | ssd_cold |
| small | 8128.8ms | 761.6ms | 1.84x | 5.8ms | 190.4ms | 60.5ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 125117.4ms | 1061.7ms | 15.85x | 8.0ms | 265.4ms | 52.7ms | ssd_cold |
| medium | 39052.2ms | 630.7ms | 6.35x | 7.2ms | 157.7ms | 48.3ms | ssd_cold |
| small | 8762.5ms | 436.0ms | 2.25x | 7.0ms | 109.0ms | 46.8ms | ssd_cold |
