# gpt-oss-20b-UD-Q6_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128 | **Profile:** dense-large (features: no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 34.8 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15381 | 45914.8ms | 15381 | 237.3ms | 9.45x | 3.0ms | 237.3ms | 40.3ms | ssd_warm | - |
| medium | 5251 | 13923.7ms | 5251 | 111.0ms | 4.04x | 2.7ms | 111.0ms | 34.9ms | ssd_warm | - |
| small | 1205 | 3771.4ms | 1205 | 69.8ms | 1.88x | 3.1ms | 69.8ms | 32.3ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
