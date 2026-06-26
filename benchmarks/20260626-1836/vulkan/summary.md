# Benchmark Results: VULKAN

**Date:** 20260626-1836 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.97x (1514.8/70.7ms, ssd_warm) | TTFT 4.02x (7669.6/83.1ms, ssd_warm) | TTFT 8.78x (22859.5/123.0ms, ssd_warm) |
| GLM-4.7-Flash-UD-Q8_K_XL | TTFT 1.49x (1513.3/177.5ms, ssd_warm) | TTFT 3.22x (7974.3/384.4ms, ssd_warm) | TTFT 9.49x (38963.3/1082.8ms, ssd_warm) |
| gpt-oss-20b-UD-Q6_K_XL | TTFT 2.01x (1111.2/38.5ms, ssd_warm) | TTFT 3.38x (3664.7/86.4ms, ssd_warm) | TTFT 7.83x (12368.8/156.3ms, ssd_warm) |
| Qwen3.6-27B-UD-Q8_K_XL | TTFT 1.29x (6596.1/418.6ms, ssd_warm) | TTFT 1.99x (22806.5/501.4ms, ssd_warm) | TTFT 3.87x (66590.4/1140.5ms, ssd_warm) |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 1.91x (2199.8/155.3ms, ssd_warm) | TTFT 3.08x (6898.9/249.7ms, ssd_warm) | TTFT 6.13x (19659.6/561.1ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 22859.5ms | 123.0ms | 8.78x | 1.3ms | 30.7ms | 42.1ms | ssd_warm |
| medium | 7669.6ms | 83.1ms | 4.02x | 1.3ms | 20.8ms | 35.0ms | ssd_warm |
| small | 1514.8ms | 70.7ms | 1.97x | 1.1ms | 17.7ms | 32.6ms | ssd_warm |

### GLM-4.7-Flash-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 38963.3ms | 1082.8ms | 9.49x | 2.5ms | 270.7ms | 103.8ms | ssd_warm |
| medium | 7974.3ms | 384.4ms | 3.22x | 1.5ms | 96.1ms | 52.8ms | ssd_warm |
| small | 1513.3ms | 177.5ms | 1.49x | 1.3ms | 44.4ms | 33.3ms | ssd_warm |

### gpt-oss-20b-UD-Q6_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 12368.8ms | 156.3ms | 7.83x | 0.8ms | 39.1ms | 32.2ms | ssd_warm |
| medium | 3664.7ms | 86.4ms | 3.38x | 0.7ms | 21.6ms | 19.8ms | ssd_warm |
| small | 1111.2ms | 38.5ms | 2.01x | 0.9ms | 9.6ms | 17.3ms | ssd_warm |

### Qwen3.6-27B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 66590.4ms | 1140.5ms | 3.87x | 4.2ms | 285.1ms | 209.3ms | ssd_warm |
| medium | 22806.5ms | 501.4ms | 1.99x | 4.2ms | 125.3ms | 183.1ms | ssd_warm |
| small | 6596.1ms | 418.6ms | 1.29x | 5.3ms | 104.6ms | 183.2ms | ssd_warm |

### Qwen3.6-35B-A3B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 19659.6ms | 561.1ms | 6.13x | 1.3ms | 140.3ms | 44.8ms | ssd_warm |
| medium | 6898.9ms | 249.7ms | 3.08x | 1.3ms | 62.4ms | 34.8ms | ssd_warm |
| small | 2199.8ms | 155.3ms | 1.91x | 1.8ms | 38.8ms | 32.4ms | ssd_warm |
