# Benchmark Results: VULKAN

**Date:** 20260810-0548 | **Context:** 131072

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004 | TTFT 1.13x (7957.8/320.0ms, ssd_warm) | TTFT 3.08x (30762.6/314.2ms, ssd_warm) | TTFT 5.54x (96976.6/344.4ms, ssd_warm) |
| GLM-4.7-Flash-UD-Q8_K_XL | TTFT 1.3x (1923.7/61.6ms, ssd_warm) | TTFT 2.92x (14084.7/95.7ms, ssd_warm) | TTFT 10.09x (85462.8/178.7ms, ssd_warm) |
| Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003 | TTFT 3.04x (4789.6/177.6ms, ssd_warm) | TTFT 3.12x (17606.8/319.6ms, ssd_warm) | TTFT 9.4x (50033.8/483.3ms, ssd_warm) |
| MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003 | TTFT 1.57x (4601.2/284.6ms, ssd_warm) | TTFT 2.89x (18336.3/944.7ms, ssd_warm) | TTFT 5.14x (60268.9/3189.3ms, ssd_warm) |
| Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002 | TTFT 1.71x (6934.7/157.0ms, ssd_warm) | TTFT 3.73x (28796.9/315.6ms, ssd_warm) | TTFT 8.65x (100548.8/1043.4ms, ssd_warm) |
| Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003 | TTFT 2.18x (2413.1/134.9ms, ssd_warm) | TTFT 2.89x (8894.3/208.0ms, ssd_warm) | TTFT 7.74x (48472.6/219.2ms, ssd_warm) |
| Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003 | TTFT 3.55x (5965.1/188.0ms, ssd_warm) | TTFT 3.05x (18763.2/220.0ms, ssd_warm) | TTFT 7.55x (52906.5/310.1ms, ssd_warm) |
| Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003 | TTFT 3.88x (4516.9/210.9ms, ssd_warm) | TTFT 6.82x (17365.3/201.7ms, ssd_warm) | TTFT 16.77x (53513.9/282.7ms, ssd_warm) |
| Qwen3.6-27B-UD-Q8_K_XL | TTFT 1.29x (8087.7/249.3ms, ssd_warm) | TTFT 2.07x (29289.3/276.3ms, ssd_warm) | TTFT 4.08x (85337.0/358.1ms, ssd_warm) |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | TTFT 2.71x (2390.3/84.0ms, ssd_warm) | TTFT 2.86x (7278.3/98.9ms, ssd_warm) | TTFT 5.74x (20502.1/137.6ms, ssd_warm) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.15x (1396.9/92.8ms, ssd_warm) | TTFT 1.58x (5280.1/102.8ms, ssd_warm) | TTFT 2.82x (16652.6/135.9ms, ssd_warm) |
| gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 | TTFT 1.41x (2623.8/100.3ms, ssd_warm) | TTFT 2.21x (8410.1/117.1ms, ssd_warm) | TTFT 4.38x (24985.6/241.1ms, ssd_warm) |
| gpt-oss-20b-UD-Q6_K_XL | TTFT 1.2x (1123.9/50.0ms, ssd_warm) | TTFT 1.63x (3576.6/63.8ms, ssd_warm) | TTFT 2.88x (11085.3/99.3ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## MoE Expert Residency (warm run)

Hit rate is the percent of expert lookups served from the in-RAM madvise
cache (vs falling through to SSD/weights). Only meaningful for MoE models
with `--moe-expert-residency` enabled; '-' otherwise.

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004 | - | - | - |
| GLM-4.7-Flash-UD-Q8_K_XL | - | - | - |
| Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003 | - | - | - |
| MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003 | - | - | - |
| Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002 | - | - | - |
| Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003 | - | - | - |
| Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003 | - | - | - |
| Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003 | - | - | - |
| Qwen3.6-27B-UD-Q8_K_XL | - | - | - |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | - | - | - |
| gemma-4-26B-A4B-it-UD-Q5_K_M | - | - | - |
| gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 | - | - | - |
| gpt-oss-20b-UD-Q6_K_XL | - | - | - |

## Per-Model Detail

### DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 96976.6ms | 344.4ms | 5.54x | 5.5ms | 344.4ms | 144.1ms | ssd_warm | - |
| medium | 30762.6ms | 314.2ms | 3.08x | 5.1ms | 314.2ms | 88.6ms | ssd_warm | - |
| small | 7957.8ms | 320.0ms | 1.13x | 6.0ms | 320.0ms | 58.6ms | ssd_warm | - |

### GLM-4.7-Flash-UD-Q8_K_XL (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 85462.8ms | 178.7ms | 10.09x | 5.5ms | 178.7ms | 71.8ms | ssd_warm | - |
| medium | 14084.7ms | 95.7ms | 2.92x | 2.7ms | 95.7ms | 55.8ms | ssd_warm | - |
| small | 1923.7ms | 61.6ms | 1.3x | 1.7ms | 61.6ms | 49.6ms | ssd_warm | - |

### Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 50033.8ms | 483.3ms | 9.4x | 2.8ms | 483.3ms | 80.7ms | ssd_warm | - |
| medium | 17606.8ms | 319.6ms | 3.12x | 2.8ms | 319.6ms | 79.9ms | ssd_warm | - |
| small | 4789.6ms | 177.6ms | 3.04x | 3.2ms | 177.6ms | 71.3ms | ssd_warm | - |

### MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 60268.9ms | 3189.3ms | 5.14x | 4.0ms | 3189.3ms | 83.5ms | ssd_warm | - |
| medium | 18336.3ms | 944.7ms | 2.89x | 3.6ms | 944.7ms | 66.8ms | ssd_warm | - |
| small | 4601.2ms | 284.6ms | 1.57x | 3.9ms | 284.6ms | 58.9ms | ssd_warm | - |

### Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 100548.8ms | 1043.4ms | 8.65x | 6.4ms | 1043.4ms | 94.3ms | ssd_warm | - |
| medium | 28796.9ms | 315.6ms | 3.73x | 5.3ms | 315.6ms | 80.0ms | ssd_warm | - |
| small | 6934.7ms | 157.0ms | 1.71x | 5.5ms | 157.0ms | 73.6ms | ssd_warm | - |

### Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 48472.6ms | 219.2ms | 7.74x | 3.1ms | 219.2ms | 52.8ms | ssd_warm | - |
| medium | 8894.3ms | 208.0ms | 2.89x | 1.6ms | 208.0ms | 50.3ms | ssd_warm | - |
| small | 2413.1ms | 134.9ms | 2.18x | 1.9ms | 134.9ms | 50.5ms | ssd_warm | - |

### Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 52906.5ms | 310.1ms | 7.55x | 3.4ms | 310.1ms | 96.9ms | ssd_warm | - |
| medium | 18763.2ms | 220.0ms | 3.05x | 3.5ms | 220.0ms | 94.8ms | ssd_warm | - |
| small | 5965.1ms | 188.0ms | 3.55x | 4.8ms | 188.0ms | 94.2ms | ssd_warm | - |

### Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 53513.9ms | 282.7ms | 16.77x | 3.4ms | 282.7ms | 34.9ms | ssd_warm | - |
| medium | 17365.3ms | 201.7ms | 6.82x | 3.2ms | 201.7ms | 30.6ms | ssd_warm | - |
| small | 4516.9ms | 210.9ms | 3.88x | 3.6ms | 210.9ms | 33.2ms | ssd_warm | - |

### Qwen3.6-27B-UD-Q8_K_XL (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 85337.0ms | 358.1ms | 4.08x | 5.4ms | 358.1ms | 213.7ms | ssd_warm | - |
| medium | 29289.3ms | 276.3ms | 2.07x | 5.4ms | 276.3ms | 210.5ms | ssd_warm | - |
| small | 8087.7ms | 249.3ms | 1.29x | 6.5ms | 249.3ms | 209.8ms | ssd_warm | - |

### Qwen3.6-35B-A3B-UD-Q8_K_XL (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 20502.1ms | 137.6ms | 5.74x | 1.3ms | 137.6ms | 59.1ms | ssd_warm | - |
| medium | 7278.3ms | 98.9ms | 2.86x | 1.3ms | 98.9ms | 57.2ms | ssd_warm | - |
| small | 2390.3ms | 84.0ms | 2.71x | 1.9ms | 84.0ms | 57.2ms | ssd_warm | - |

### gemma-4-26B-A4B-it-UD-Q5_K_M (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 16652.6ms | 135.9ms | 2.82x | 1.0ms | 135.9ms | 71.5ms | ssd_warm | - |
| medium | 5280.1ms | 102.8ms | 1.58x | 0.9ms | 102.8ms | 68.9ms | ssd_warm | - |
| small | 1396.9ms | 92.8ms | 1.15x | 1.0ms | 92.8ms | 68.3ms | ssd_warm | - |

### gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 24985.6ms | 241.1ms | 4.38x | 1.6ms | 241.1ms | 55.0ms | ssd_warm | - |
| medium | 8410.1ms | 117.1ms | 2.21x | 1.6ms | 117.1ms | 52.1ms | ssd_warm | - |
| small | 2623.8ms | 100.3ms | 1.41x | 2.2ms | 100.3ms | 51.4ms | ssd_warm | - |

### gpt-oss-20b-UD-Q6_K_XL (profile: default)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 11085.3ms | 99.3ms | 2.88x | 0.7ms | 99.3ms | 44.9ms | ssd_warm | - |
| medium | 3576.6ms | 63.8ms | 1.63x | 0.7ms | 63.8ms | 43.2ms | ssd_warm | - |
| small | 1123.9ms | 50.0ms | 1.2x | 0.9ms | 50.0ms | 42.1ms | ssd_warm | - |
