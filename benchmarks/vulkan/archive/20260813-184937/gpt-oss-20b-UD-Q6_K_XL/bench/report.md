# gpt-oss-20b-UD-Q6_K_XL -- llama-bench (vulkan)

**Repetitions per test:** 5  
**Model params:** 20.9B  
**Model size on disk:** 11.2 GiB  

## Prompt Processing (tg=0, pure prefill)

| pp tokens | avg ms | pp t/s | sd (t/s) |
|----------:|-------:|-------:|---------:|
| 512 | 320.7 | 1596.7 | 14.1 |
| 2048 | 1272.5 | 1609.4 | 2.2 |
| 8192 | 5529.6 | 1481.5 | 3.2 |
| 16384 | 12356.4 | 1326.0 | 1.8 |

## Text Generation (pp=0, pure generation)

| tg tokens | avg ms | tg t/s | sd (t/s) |
|----------:|-------:|-------:|---------:|
| 128 | 1696.3 | 75.5 | 0.8 |
| 256 | 3405.2 | 75.2 | 0.1 |
| 512 | 6798.1 | 75.3 | 0.4 |
| 1024 | 13700.8 | 74.7 | 0.2 |
| 2048 | 27512.2 | 74.4 | 0.3 |

_Source: [`raw.json`](./raw.json)_
