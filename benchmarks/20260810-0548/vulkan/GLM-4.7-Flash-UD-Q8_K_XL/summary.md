# GLM-4.7-Flash-UD-Q8_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 62.9 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15489 | 85462.8ms | 15489 | 178.7ms | 10.09x | 5.5ms | 178.7ms | 71.8ms | ssd_warm | - |
| medium | 5237 | 14084.7ms | 5237 | 95.7ms | 2.92x | 2.7ms | 95.7ms | 55.8ms | ssd_warm | - |
| small | 1145 | 1923.7ms | 1145 | 61.6ms | 1.3x | 1.7ms | 61.6ms | 49.6ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
