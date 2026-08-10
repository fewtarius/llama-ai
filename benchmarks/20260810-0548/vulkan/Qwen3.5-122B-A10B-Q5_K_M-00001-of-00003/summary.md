# Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 493.7 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 52906.5ms | 15723 | 310.1ms | 7.55x | 3.4ms | 310.1ms | 96.9ms | ssd_warm | - |
| medium | 5411 | 18763.2ms | 5411 | 220.0ms | 3.05x | 3.5ms | 220.0ms | 94.8ms | ssd_warm | - |
| small | 1244 | 5965.1ms | 1244 | 188.0ms | 3.55x | 4.8ms | 188.0ms | 94.2ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
