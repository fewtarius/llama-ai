# Qwen3.6-35B-A3B-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 234.6 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 20502.1ms | 15723 | 137.6ms | 5.74x | 1.3ms | 137.6ms | 59.1ms | ssd_warm | - |
| medium | 5411 | 7278.3ms | 5411 | 98.9ms | 2.86x | 1.3ms | 98.9ms | 57.2ms | ssd_warm | - |
| small | 1244 | 2390.3ms | 1244 | 84.0ms | 2.71x | 1.9ms | 84.0ms | 57.2ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
