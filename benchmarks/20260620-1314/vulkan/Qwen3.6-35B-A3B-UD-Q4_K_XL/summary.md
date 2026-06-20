# Qwen3.6-35B-A3B-UD-Q4_K_XL (vulkan)

**Context:** 196608 | **Output tokens/req:** 128

## Prompt Cache Performance

| Size | Cold Tok | Cold TTFT | Warm Tok | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cache |
|------|----------|-----------|----------|-----------|-------------|-------------|-------------|------------|-------|
| large | 15721 | 22191.8ms | 15721 | 2289.0ms | 5.13x | 1.4ms | 381.5ms | 19.7ms | miss |
| medium | 5409 | 7090.4ms | 5409 | 297.5ms | 3.52x | 1.3ms | 49.6ms | 18.6ms | miss |
| small | 1243 | 1847.6ms | 1243 | 245.2ms | 1.63x | 1.5ms | 40.9ms | 18.3ms | miss |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval time)
**TTFT Speedup** = cold TTFT / warm TTFT
