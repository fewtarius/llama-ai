# Benchmark Results: VULKAN

**Date:** 20260626-0648 | **Context:** ?

## TTFT Speedup by Size

| Model | Small (1K tok) | Medium (5K tok) | Large (15K tok) |
|-------|---------------|-----------------|-----------------|
| Qwen3.6-27B-UD-Q8_K_XL | TTFT 1.15x (5360.4/850.6ms, ssd_cold) | TTFT 1.95x (21052.4/1242.4ms, ssd_cold) | TTFT 3.37x (63696.1/2198.5ms, ssd_cold) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better)
**Format:** TTFT speedup (cold_ms/warm_ms, cache_state)
**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory cache, `miss` = no hit

## Per-Model Detail

### Qwen3.6-27B-UD-Q8_K_XL

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|-----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 63696.1ms | 2198.5ms | 3.37x | 4.1ms | 549.6ms | 167.2ms | ssd_cold |
| medium | 21052.4ms | 1242.4ms | 1.95x | 3.9ms | 310.6ms | 163.8ms | ssd_cold |
| small | 5360.4ms | 850.6ms | 1.15x | 4.3ms | 212.7ms | 165.2ms | ssd_cold |
