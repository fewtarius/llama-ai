# Benchmark Results: VULKAN

**Date:** 20260721-1845 | **Context:** 65536

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | TTFT 2.09x (7348.4/262.1ms, ssd_warm) | TTFT 5.99x (56168.2/620.9ms, ssd_warm) | TTFT 15.83x (339087.7/1502.7ms, ssd_warm) |
| Qwen3.6-27B-UD-Q4_K_XL | TTFT 1.69x (19598.1/527.2ms, ssd_warm) | TTFT 4.0x (84626.3/692.4ms, ssd_warm) | TTFT 9.85x (262083.2/1020.7ms, ssd_warm) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | TTFT 1.92x (6210.7/217.1ms, ssd_warm) | TTFT 4.94x (26730.8/284.6ms, ssd_warm) | TTFT 11.92x (83739.5/456.2ms, ssd_warm) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | TTFT 3.18x (7345.9/288.1ms, ssd_warm) | TTFT 5.56x (30284.2/382.7ms, ssd_warm) | TTFT 11.56x (94644.6/565.8ms, ssd_warm) |
| gpt-oss-20b-UD-Q6_K_XL | TTFT 1.95x (4510.9/134.7ms, ssd_warm) | TTFT 4.63x (18181.3/196.6ms, ssd_warm) | TTFT 11.34x (58805.3/222.4ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## MoE Expert Residency (warm run)

Hit rate is the percent of expert lookups served from the in-RAM madvise
cache (vs falling through to SSD/weights). Only meaningful for MoE models
with `--moe-expert-residency` enabled; '-' otherwise.

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| GLM-4.7-Flash-Q4_K_M | 89.5% (21100h/2476m) | 89.6% (21129h/2447m) | 89.5% (21104h/2472m) |
| Qwen3.6-27B-UD-Q4_K_XL | - | - | - |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 57.2% (24325h/18235m) | 60.5% (25766h/16794m) | 58.0% (24704h/17856m) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | 68.3% (3665h/1703m) | 62.3% (3343h/2025m) | 65.4% (3635h/1925m) |
| gpt-oss-20b-UD-Q6_K_XL | 97.1% (11715h/345m) | 97.5% (11759h/301m) | 97.5% (11759h/301m) |

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M (profile: moe-optimized)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 339087.7ms | 1502.7ms | 15.83x | 21.9ms | 375.7ms | 165.4ms | ssd_warm | 89.5% |
| medium | 56168.2ms | 620.9ms | 5.99x | 10.7ms | 155.2ms | 80.7ms | ssd_warm | 89.6% |
| small | 7348.4ms | 262.1ms | 2.09x | 6.4ms | 65.5ms | 48.8ms | ssd_warm | 89.5% |

### Qwen3.6-27B-UD-Q4_K_XL (profile: dense-large)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 262083.2ms | 1020.7ms | 9.85x | 16.7ms | 255.2ms | 222.0ms | ssd_warm | - |
| medium | 84626.3ms | 692.4ms | 4.0x | 15.6ms | 173.1ms | 211.7ms | ssd_warm | - |
| small | 19598.1ms | 527.2ms | 1.69x | 15.8ms | 131.8ms | 205.2ms | ssd_warm | - |

### Qwen3.6-35B-A3B-UD-Q4_K_XL (profile: moe-optimized)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 83739.5ms | 456.2ms | 11.92x | 5.3ms | 114.0ms | 53.2ms | ssd_warm | 58.0% |
| medium | 26730.8ms | 284.6ms | 4.94x | 4.9ms | 71.2ms | 49.0ms | ssd_warm | 60.5% |
| small | 6210.7ms | 217.1ms | 1.92x | 5.0ms | 54.3ms | 47.2ms | ssd_warm | 57.2% |

### gemma-4-26B-A4B-it-UD-Q5_K_M (profile: moe-optimized)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 94644.6ms | 565.8ms | 11.56x | 5.5ms | 141.5ms | 165.0ms | ssd_warm | 65.4% |
| medium | 30284.2ms | 382.7ms | 5.56x | 5.0ms | 95.7ms | 159.8ms | ssd_warm | 62.3% |
| small | 7345.9ms | 288.1ms | 3.18x | 5.2ms | 72.0ms | 153.8ms | ssd_warm | 68.3% |

### gpt-oss-20b-UD-Q6_K_XL (profile: moe-optimized)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 58805.3ms | 222.4ms | 11.34x | 3.8ms | 55.6ms | 43.3ms | ssd_warm | 97.5% |
| medium | 18181.3ms | 196.6ms | 4.63x | 3.5ms | 49.2ms | 37.2ms | ssd_warm | 97.5% |
| small | 4510.9ms | 134.7ms | 1.95x | 3.7ms | 33.7ms | 34.8ms | ssd_warm | 97.1% |
