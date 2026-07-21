# gpt-oss-20b-UD-Q6_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** moe-optimized (features: moe-residency, no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 34.8 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15381 | 58805.3ms | 15381 | 222.4ms | 11.34x | 3.8ms | 55.6ms | 43.3ms | ssd_warm | 97.5% |
| medium | 5251 | 18181.3ms | 5251 | 196.6ms | 4.63x | 3.5ms | 49.2ms | 37.2ms | ssd_warm | 97.5% |
| small | 1205 | 4510.9ms | 1205 | 134.7ms | 1.95x | 3.7ms | 33.7ms | 34.8ms | ssd_warm | 97.1% |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
