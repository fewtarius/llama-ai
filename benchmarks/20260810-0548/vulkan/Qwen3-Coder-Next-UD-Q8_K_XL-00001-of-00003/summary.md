# Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 272.9 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15664 | 48472.6ms | 15664 | 219.2ms | 7.74x | 3.1ms | 219.2ms | 52.8ms | ssd_warm | - |
| medium | 5399 | 8894.3ms | 5399 | 208.0ms | 2.89x | 1.6ms | 208.0ms | 50.3ms | ssd_warm | - |
| small | 1252 | 2413.1ms | 1252 | 134.9ms | 2.18x | 1.9ms | 134.9ms | 50.5ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
