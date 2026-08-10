# gpt-oss-20b-UD-Q6_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 34.8 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15381 | 11085.3ms | 15381 | 99.3ms | 2.88x | 0.7ms | 99.3ms | 44.9ms | ssd_warm | - |
| medium | 5251 | 3576.6ms | 5251 | 63.8ms | 1.63x | 0.7ms | 63.8ms | 43.2ms | ssd_warm | - |
| small | 1205 | 1123.9ms | 1205 | 50.0ms | 1.2x | 0.9ms | 50.0ms | 42.1ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
