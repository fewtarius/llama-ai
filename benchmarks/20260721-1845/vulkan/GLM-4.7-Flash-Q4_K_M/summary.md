# GLM-4.7-Flash-Q4_K_M (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** moe-optimized (features: moe-residency, no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 62.9 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15489 | 339087.7ms | 15489 | 1502.7ms | 15.83x | 21.9ms | 375.7ms | 165.4ms | ssd_warm | 89.5% |
| medium | 5237 | 56168.2ms | 5237 | 620.9ms | 5.99x | 10.7ms | 155.2ms | 80.7ms | ssd_warm | 89.6% |
| small | 1145 | 7348.4ms | 1145 | 262.1ms | 2.09x | 6.4ms | 65.5ms | 48.8ms | ssd_warm | 89.5% |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
