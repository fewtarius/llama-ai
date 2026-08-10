# MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 451.6 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15080 | 60268.9ms | 15080 | 3189.3ms | 5.14x | 4.0ms | 3189.3ms | 83.5ms | ssd_warm | - |
| medium | 5157 | 18336.3ms | 5157 | 944.7ms | 2.89x | 3.6ms | 944.7ms | 66.8ms | ssd_warm | - |
| small | 1170 | 4601.2ms | 1170 | 284.6ms | 1.57x | 3.9ms | 284.6ms | 58.9ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
