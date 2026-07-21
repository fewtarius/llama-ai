# gemma-4-26B-A4B-it-UD-Q5_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** moe-optimized (features: moe-residency, no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 242.9 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 17349 | 94644.6ms | 17349 | 565.8ms | 11.56x | 5.5ms | 141.5ms | 165.0ms | ssd_warm | 65.4% |
| medium | 6085 | 30284.2ms | 6085 | 382.7ms | 5.56x | 5.0ms | 95.7ms | 159.8ms | ssd_warm | 62.3% |
| small | 1414 | 7345.9ms | 1414 | 288.1ms | 3.18x | 5.2ms | 72.0ms | 153.8ms | ssd_warm | 68.3% |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
