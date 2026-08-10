# Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 262.2 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 18154 | 50033.8ms | 18154 | 483.3ms | 9.4x | 2.8ms | 483.3ms | 80.7ms | ssd_warm | - |
| medium | 6346 | 17606.8ms | 6346 | 319.6ms | 3.12x | 2.8ms | 319.6ms | 79.9ms | ssd_warm | - |
| small | 1490 | 4789.6ms | 1490 | 177.6ms | 3.04x | 3.2ms | 177.6ms | 71.3ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
