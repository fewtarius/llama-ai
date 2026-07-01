# Benchmark Results: VULKAN

**Date:** 20260630-1942 | **Context:** 196608

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 2.3x (1529.1/91.8ms, ssd_warm) | TTFT 3.96x (5898.7/106.2ms, ssd_warm) | TTFT 9.63x (23641.2/165.3ms, ssd_warm) |
| GLM-4.7-Flash-UD-Q8_K_XL | TTFT 1.59x (1543.3/126.3ms, ssd_warm) | TTFT 3.49x (8197.1/320.5ms, ssd_warm) | TTFT 9.3x (40592.1/1072.6ms, ssd_warm) |
| gpt-oss-20b-UD-Q6_K_XL | TTFT 1.74x (1036.6/36.8ms, ssd_warm) | TTFT 3.48x (3773.9/61.9ms, ssd_warm) | TTFT 8.28x (12749.1/165.6ms, ssd_warm) |
| Qwen3.6-27B-UD-Q8_K_XL | TTFT 1.0x (4623.6/4613.5ms, miss) | TTFT 1.0x (20324.1/20297.6ms, miss) | TTFT 1.0x (62709.6/62730.3ms, miss) |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.89x (2254.5/140.5ms, ssd_warm) | TTFT 3.17x (6923.4/208.0ms, ssd_warm) | TTFT 6.73x (20285.2/472.0ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 23641.2ms | 165.3ms | 9.63x | 1.4ms | 41.3ms | 65.3ms | ssd_warm |
| medium | 5898.7ms | 106.2ms | 3.96x | 1.0ms | 26.5ms | 49.1ms | ssd_warm |
| small | 1529.1ms | 91.8ms | 2.3x | 1.1ms | 23.0ms | 41.7ms | ssd_warm |

### GLM-4.7-Flash-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 40592.1ms | 1072.6ms | 9.3x | 2.6ms | 268.1ms | 111.1ms | ssd_warm |
| medium | 8197.1ms | 320.5ms | 3.49x | 1.6ms | 80.1ms | 53.6ms | ssd_warm |
| small | 1543.3ms | 126.3ms | 1.59x | 1.3ms | 31.6ms | 34.1ms | ssd_warm |

### gpt-oss-20b-UD-Q6_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 12749.1ms | 165.6ms | 8.28x | 0.8ms | 41.4ms | 34.3ms | ssd_warm |
| medium | 3773.9ms | 61.9ms | 3.48x | 0.7ms | 15.5ms | 19.3ms | ssd_warm |
| small | 1036.6ms | 36.8ms | 1.74x | 0.9ms | 9.2ms | 15.0ms | ssd_warm |

### Qwen3.6-27B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 62709.6ms | 62730.3ms | 1.0x | 4.0ms | 4.0ms | 163.6ms | miss |
| medium | 20324.1ms | 20297.6ms | 1.0x | 3.8ms | 3.8ms | 162.4ms | miss |
| small | 4623.6ms | 4613.5ms | 1.0x | 3.7ms | 3.7ms | 167.2ms | miss |

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 20285.2ms | 472.0ms | 6.73x | 1.3ms | 118.0ms | 49.2ms | ssd_warm |
| medium | 6923.4ms | 208.0ms | 3.17x | 1.3ms | 52.0ms | 34.0ms | ssd_warm |
| small | 2254.5ms | 140.5ms | 1.89x | 1.8ms | 35.1ms | 32.4ms | ssd_warm |
