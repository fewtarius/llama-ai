# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 196608 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 242.9 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 17349 | 16652.6ms | 17349 | 135.9ms | 2.82x | 1.0ms | 135.9ms | 71.5ms | ssd_warm | - |
| medium | 6085 | 5280.1ms | 6085 | 102.8ms | 1.58x | 0.9ms | 102.8ms | 68.9ms | ssd_warm | - |
| small | 1414 | 1396.9ms | 1414 | 92.8ms | 1.15x | 1.0ms | 92.8ms | 68.3ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
