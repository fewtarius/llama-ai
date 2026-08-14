# gpt-oss-20b-UD-Q6_K_XL -- llama-batched-bench (vulkan)

**Per-test prompt:** 2048 tokens prefill, 128 tokens generate

## Concurrent Throughput

| Parallel | Total t/s | PP t/s | TG t/s | Per-slot total t/s |
|---------:|----------:|-------:|-------:|-------------------:|
| 1 | 720.8 | 1617.1 | 73.0 | 720.8 |
| 2 | 904.3 | 1624.5 | 111.7 | 452.2 |
| 4 | 1065.2 | 1642.3 | 160.8 | 266.3 |

_Each row aggregates `n_parallel` simultaneous prompts through the same KV cache._
_Source: [`raw.json`](./raw.json)_
