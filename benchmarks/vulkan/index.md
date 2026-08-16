# VULKAN Benchmark Leaderboard

**Models tested:** 1

## Performance at a Glance

| Model | tg t/s (128) | tg t/s (512) | pp t/s (2k) | pp t/s (8k) | warm 15k TTFT | cache speedup |
|-------|------------:|------------:|------------:|------------:|--------------:|--------------:|
| [gpt-oss-20b-UD-Q6_K_XL](gpt-oss-20b-UD-Q6_K_XL/summary.md) | 0.0 | 0.0 | 0.0 | 0.0 | - | - |

**tg t/s (N)** = text generation throughput with N tokens generated (llama-bench, pp=512).  
**pp t/s (N)** = prompt processing throughput with N-token prefill (llama-bench, tg=128).  
**warm 15k TTFT** = server-side prompt eval after restoring SSD cache for a 15K-token prompt.  
**cache speedup** = cold TTFT / warm TTFT at 15K.
