# Qwen3.6-27B-UD-Q8_K_XL (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 572.8 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 85337.0ms | 15723 | 358.1ms | 4.08x | 5.4ms | 358.1ms | 213.7ms | ssd_warm | - |
| medium | 5411 | 29289.3ms | 5411 | 276.3ms | 2.07x | 5.4ms | 276.3ms | 210.5ms | ssd_warm | - |
| small | 1244 | 8087.7ms | 1244 | 249.3ms | 1.29x | 6.5ms | 249.3ms | 209.8ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
