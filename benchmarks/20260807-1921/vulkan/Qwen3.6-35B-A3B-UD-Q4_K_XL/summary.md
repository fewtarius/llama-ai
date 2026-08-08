# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** dense-large (features: no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 227.2 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 70885.2ms | 15723 | 377.3ms | 23.55x | 4.5ms | 377.3ms | 48.3ms | ssd_warm | - |
| medium | 5411 | 22187.8ms | 5411 | 227.2ms | 7.65x | 4.1ms | 227.2ms | 44.1ms | ssd_warm | - |
| small | 1244 | 5316.2ms | 1244 | 150.8ms | 5.56x | 4.3ms | 150.8ms | 42.6ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
