# Qwen3.6-35B-A3B-UD-Q5_K_XL (vulkan)

**Context:** 65536 | **Output tokens/req:** 128 | **Profile:** moe-optimized (features: moe-residency, no-checkpoint-near-end, kv-unified, cpu-moe)
**SSD cache footprint after run:** 227.2 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 304832.7ms | 15723 | 842.1ms | 15.69x | 19.4ms | 210.5ms | 119.9ms | ssd_warm | 88.8% |
| medium | 5411 | 110832.7ms | 5411 | 643.0ms | 7.19x | 20.5ms | 160.7ms | 116.7ms | ssd_warm | 88.7% |
| small | 1244 | 31440.2ms | 1244 | 476.1ms | 10.0x | 25.3ms | 119.0ms | 926.9ms | ssd_warm | 99.3% |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
