# gpt-oss-120b-UD-Q8_K_XL-00001-of-00002 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 52.2 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15381 | 24985.6ms | 15381 | 241.1ms | 4.38x | 1.6ms | 241.1ms | 55.0ms | ssd_warm | - |
| medium | 5251 | 8410.1ms | 5251 | 117.1ms | 2.21x | 1.6ms | 117.1ms | 52.1ms | ssd_warm | - |
| small | 1205 | 2623.8ms | 1205 | 100.3ms | 1.41x | 2.2ms | 100.3ms | 51.4ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
