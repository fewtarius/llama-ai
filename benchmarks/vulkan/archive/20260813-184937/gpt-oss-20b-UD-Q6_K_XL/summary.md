# gpt-oss-20b-UD-Q6_K_XL

**Backend:** vulkan

**Profile:** default (features: kv-unified)

## Cache Performance

[Drilldown](./cache/analysis.md)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | MoE Hit | Cache |
|------|----------:|----------:|-------------:|---------|-------|
| large | 10234.5ms | 110.2ms | 2.61x | - | ssd_warm |
| medium | 3404.4ms | 69.8ms | 1.54x | - | ssd_warm |
| small | 997.8ms | 55.2ms | 1.19x | - | ssd_warm |

## llama-bench

[Drilldown](./bench/report.md)

### Prompt Processing (tg=128)

| pp tokens | pp t/s |
|----------:|-------:|
| 512 | 1596.7 |
| 2048 | 1609.4 |
| 8192 | 1481.5 |
| 16384 | 1326.0 |

### Text Generation (pp=512)

| tg tokens | tg t/s |
|----------:|-------:|
| 128 | 75.5 |
| 256 | 75.2 |
| 512 | 75.3 |
| 1024 | 74.7 |
| 2048 | 74.4 |

## llama-batched-bench

[Drilldown](./batched/report.md)

| Parallel prompts | Total t/s |
|-----------------:|----------:|
| 1 | 720.8 |
| 2 | 904.3 |
| 4 | 1065.2 |

## Files

- [`cache/analysis.md`](./cache/analysis.md) -- cache drilldown
- [`bench/report.md`](./bench/report.md) -- llama-bench drilldown
- [`batched/report.md`](./batched/report.md) -- batched drilldown
- [`summary.json`](./summary.json) -- machine-readable aggregate