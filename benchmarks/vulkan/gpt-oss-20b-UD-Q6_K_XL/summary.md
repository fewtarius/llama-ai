# gpt-oss-20b-UD-Q6_K_XL

**Backend:** vulkan

**Profile:** default (features: kv-unified)

## Cache Performance

[Drilldown](./cache/analysis.md)

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | MoE Hit | Cache |
|------|----------:|----------:|-------------:|---------|-------|
| large | 17071.4ms | 184.4ms | 2.52x | - | ssd_warm |
| medium | 5902.1ms | 118.6ms | 1.57x | - | ssd_warm |
| small | 1798.3ms | 91.8ms | 1.19x | - | ssd_warm |

## Files

- [`cache/analysis.md`](./cache/analysis.md) -- cache drilldown
- [`summary.json`](./summary.json) -- machine-readable aggregate