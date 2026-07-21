# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** moe-optimized (features: moe-residency, no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 227.2 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 83739.5ms | 15723 | 456.2ms | 11.92x | 5.3ms | 114.0ms | 53.2ms | ssd_warm | 58.0% |
| medium | 5411 | 26730.8ms | 5411 | 284.6ms | 4.94x | 4.9ms | 71.2ms | 49.0ms | ssd_warm | 60.5% |
| small | 1244 | 6210.7ms | 1244 | 217.1ms | 1.92x | 5.0ms | 54.3ms | 47.2ms | ssd_warm | 57.2% |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
