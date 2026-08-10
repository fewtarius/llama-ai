# Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 501.1 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 53513.9ms | 15723 | 282.7ms | 16.77x | 3.4ms | 282.7ms | 34.9ms | ssd_warm | - |
| medium | 5411 | 17365.3ms | 5411 | 201.7ms | 6.82x | 3.2ms | 201.7ms | 30.6ms | ssd_warm | - |
| small | 1244 | 4516.9ms | 1244 | 210.9ms | 3.88x | 3.6ms | 210.9ms | 33.2ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
