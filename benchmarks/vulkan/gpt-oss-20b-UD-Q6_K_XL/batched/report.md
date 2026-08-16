# gpt-oss-20b-UD-Q6_K_XL -- llama-batched-bench (vulkan)

**Per-test prompt:** 2048 tokens prefill, 128 tokens generate

## Concurrent Throughput

| Parallel | Total t/s | PP t/s | TG t/s | Per-slot total t/s |
|---------:|----------:|-------:|-------:|-------------------:|
| 1 | 685.8 | 1508.6 | 70.5 | 685.8 |
| 2 | 873.2 | 1571.5 | 107.7 | 436.6 |
| 4 | 1035.1 | 1587.5 | 157.6 | 258.8 |
| 8 | 1135.2 | 1592.6 | 202.9 | 141.9 |

_Each row aggregates `n_parallel` simultaneous prompts through the same KV cache._
_Source: [`raw.json`](./raw.json)_
