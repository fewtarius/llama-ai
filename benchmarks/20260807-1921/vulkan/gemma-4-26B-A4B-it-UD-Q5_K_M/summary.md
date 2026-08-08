# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** dense-large (features: no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 242.9 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 17349 | 63787.1ms | 17349 | 461.0ms | 7.93x | 3.7ms | 461.0ms | 69.5ms | ssd_warm | - |
| medium | 6085 | 19424.1ms | 6085 | 276.0ms | 3.33x | 3.2ms | 276.0ms | 63.2ms | ssd_warm | - |
| small | 1414 | 4731.8ms | 1414 | 222.3ms | 1.6x | 3.3ms | 222.3ms | 61.4ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
