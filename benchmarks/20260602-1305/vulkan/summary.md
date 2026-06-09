# Benchmark Results: VULKAN

**Date:** 20260602-1305 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 3.32x (34330.6/536.8ms, ssd_cold) | TTFT 20.05x (607780.2/1605.0ms, ssd_cold) | TTFT 0x (0/0ms, miss) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.95x (15236.6/1016.8ms, ssd_cold) | TTFT 5.75x (77459.2/1459.2ms, ssd_cold) | TTFT 18.8x (315506.0/1869.7ms, ssd_cold) |
| Qwen3.5-27B-UD-Q5_K_XL | TTFT 1.86x (55567.0/1857.6ms, ssd_cold) | TTFT 4.77x (250271.8/2530.9ms, ssd_cold) | TTFT 13.34x (875971.3/4114.1ms, ssd_cold) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 2.22x (15344.3/595.3ms, ssd_cold) | TTFT 6.84x (75723.0/812.3ms, ssd_cold) | TTFT 19.19x (270568.5/1382.2ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 0ms | 0ms | 0x | 0ms | 0ms | 0ms | miss |
| medium | 607780.2ms | 1605.0ms | 20.05x | 116.1ms | 401.2ms | 228.1ms | ssd_cold |
| small | 34330.6ms | 536.8ms | 3.32x | 30.0ms | 134.2ms | 108.2ms | ssd_cold |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 315506.0ms | 1869.7ms | 18.8x | 18.2ms | 467.4ms | 121.1ms | ssd_cold |
| medium | 77459.2ms | 1459.2ms | 5.75x | 12.7ms | 364.8ms | 113.2ms | ssd_cold |
| small | 15236.6ms | 1016.8ms | 1.95x | 10.8ms | 254.2ms | 107.3ms | ssd_cold |

### Qwen3.5-27B-UD-Q5_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 875971.3ms | 4114.1ms | 13.34x | 55.7ms | 1028.5ms | 518.3ms | ssd_cold |
| medium | 250271.8ms | 2530.9ms | 4.77x | 46.3ms | 632.7ms | 489.8ms | ssd_cold |
| small | 55567.0ms | 1857.6ms | 1.86x | 44.7ms | 464.4ms | 480.6ms | ssd_cold |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 270568.5ms | 1382.2ms | 19.19x | 17.2ms | 345.5ms | 102.2ms | ssd_cold |
| medium | 75723.0ms | 812.3ms | 6.84x | 14.0ms | 203.1ms | 92.2ms | ssd_cold |
| small | 15344.3ms | 595.3ms | 2.22x | 12.3ms | 148.8ms | 86.7ms | ssd_cold |
