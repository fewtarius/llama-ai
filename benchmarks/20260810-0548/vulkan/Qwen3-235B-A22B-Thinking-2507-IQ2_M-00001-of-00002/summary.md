# Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002 (vulkan)

**Context:** 131072 | **Output tokens/req:** 128 | **Profile:** default (features: kv-unified)
**SSD cache footprint after run:** 244.7 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15666 | 100548.8ms | 15666 | 1043.4ms | 8.65x | 6.4ms | 1043.4ms | 94.3ms | ssd_warm | - |
| medium | 5401 | 28796.9ms | 5401 | 315.6ms | 3.73x | 5.3ms | 315.6ms | 80.0ms | ssd_warm | - |
| small | 1254 | 6934.7ms | 1254 | 157.0ms | 1.71x | 5.5ms | 157.0ms | 73.6ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
