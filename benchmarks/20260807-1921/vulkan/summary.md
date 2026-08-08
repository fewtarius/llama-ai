# Benchmark Results: VULKAN

**Date:** 20260807-1921 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 1.6x (4731.8/222.3ms, ssd_warm) | TTFT 3.33x (19424.1/276.0ms, ssd_warm) | TTFT 7.93x (63787.1/461.0ms, ssd_warm) |
| gpt-oss-20b-UD-Q6_K_XL | TTFT 1.88x (3771.4/69.8ms, ssd_warm) | TTFT 4.04x (13923.7/111.0ms, ssd_warm) | TTFT 9.45x (45914.8/237.3ms, ssd_warm) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 5.56x (5316.2/150.8ms, ssd_warm) | TTFT 7.65x (22187.8/227.2ms, ssd_warm) | TTFT 23.55x (70885.2/377.3ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## MoE Expert Residency (warm run)

Hit rate is the percent of expert lookups served from the in-RAM madvise
cache (vs falling through to SSD/weights). Only meaningful for MoE models
with `--moe-expert-residency` enabled; '-' otherwise.

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| gemma-4-26B-A4B-it-UD-Q5_K_M | - | - | - |
| gpt-oss-20b-UD-Q6_K_XL | - | - | - |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | - | - | - |

## Per-Model Detail

### gemma-4-26B-A4B-it-UD-Q5_K_M (profile: dense-large)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 63787.1ms | 461.0ms | 7.93x | 3.7ms | 461.0ms | 69.5ms | ssd_warm | - |
| medium | 19424.1ms | 276.0ms | 3.33x | 3.2ms | 276.0ms | 63.2ms | ssd_warm | - |
| small | 4731.8ms | 222.3ms | 1.6x | 3.3ms | 222.3ms | 61.4ms | ssd_warm | - |

### gpt-oss-20b-UD-Q6_K_XL (profile: dense-large)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 45914.8ms | 237.3ms | 9.45x | 3.0ms | 237.3ms | 40.3ms | ssd_warm | - |
| medium | 13923.7ms | 111.0ms | 4.04x | 2.7ms | 111.0ms | 34.9ms | ssd_warm | - |
| small | 3771.4ms | 69.8ms | 1.88x | 3.1ms | 69.8ms | 32.3ms | ssd_warm | - |

### Qwen3.6-35B-A3B-UD-Q4_K_XL (profile: dense-large)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 70885.2ms | 377.3ms | 23.55x | 4.5ms | 377.3ms | 48.3ms | ssd_warm | - |
| medium | 22187.8ms | 227.2ms | 7.65x | 4.1ms | 227.2ms | 44.1ms | ssd_warm | - |
| small | 5316.2ms | 150.8ms | 5.56x | 4.3ms | 150.8ms | 42.6ms | ssd_warm | - |
