# DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 59.8 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 17579 | 96976.6ms | 17579 | 344.4ms | 5.54x | 5.5ms | 344.4ms | 144.1ms | ssd_warm | - |
| medium | 6052 | 30762.6ms | 6052 | 314.2ms | 3.08x | 5.1ms | 314.2ms | 88.6ms | ssd_warm | - |
| small | 1333 | 7957.8ms | 1333 | 320.0ms | 1.13x | 6.0ms | 320.0ms | 58.6ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
