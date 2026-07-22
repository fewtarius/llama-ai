# Benchmark Results: VULKAN

**Date:** 20260721-2041 | **Context:** 65536

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q5_K_XL | TTFT 10.0x (31440.2/476.1ms, ssd_warm) | TTFT 7.19x (110832.7/643.0ms, ssd_warm) | TTFT 15.69x (304832.7/842.1ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## MoE Expert Residency (warm run)

Hit rate is the percent of expert lookups served from the in-RAM madvise
cache (vs falling through to SSD/weights). Only meaningful for MoE models
with `--moe-expert-residency` enabled; '-' otherwise.

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-35B-A3B-UD-Q5_K_XL | 99.3% (36248h/264m) | 88.7% (43019h/5493m) | 88.8% (43077h/5435m) |

## Per-Model Detail

### Qwen3.6-35B-A3B-UD-Q5_K_XL (profile: moe-optimized)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 304832.7ms | 842.1ms | 15.69x | 19.4ms | 210.5ms | 119.9ms | ssd_warm | 88.8% |
| medium | 110832.7ms | 643.0ms | 7.19x | 20.5ms | 160.7ms | 116.7ms | ssd_warm | 88.7% |
| small | 31440.2ms | 476.1ms | 10.0x | 25.3ms | 119.0ms | 926.9ms | ssd_warm | 99.3% |
