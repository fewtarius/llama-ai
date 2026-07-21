# Qwen3.6-27B-UD-Q4_K_XL (vulkan)

**Context:** 32768 | **Output tokens/req:** 128 | **Profile:** dense-large (features: no-checkpoint-near-end, kv-unified)
**SSD cache footprint after run:** 514.5 MiB

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache | MoE hit |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|---------|
| large | 15723 | 262083.2ms | 15723 | 1020.7ms | 9.85x | 16.7ms | 255.2ms | 222.0ms | ssd_warm | - |
| medium | 5411 | 84626.3ms | 5411 | 692.4ms | 4.0x | 15.6ms | 173.1ms | 211.7ms | ssd_warm | - |
| small | 1244 | 19598.1ms | 1244 | 527.2ms | 1.69x | 15.8ms | 131.8ms | 205.2ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE models with --moe-expert-residency enabled)
