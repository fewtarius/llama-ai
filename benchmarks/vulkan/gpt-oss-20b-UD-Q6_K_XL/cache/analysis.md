# gpt-oss-20b-UD-Q6_K_XL — Cache Test (vulkan)

## Warm Run: large prompt

# Unknown — Cache Run Analysis

**Tasks captured:** 1 (long-context >5000 tokens: 0, short-context: 1)

## Performance by Workload Type

| Metric | Long Context | Short Context | Overall |
| :--- | :---: | :---: | :---: |
| **Prompt Processing** |  N/A t/s |  5.4 t/s |  5.4 t/s |
| **Decode (Generation)** |  N/A t/s |  11.8 t/s |  11.8 t/s |

## Decode Speed Distribution

- **p50 decode:** 11.8 t/s
- **p95 decode:** 11.8 t/s

## Decode Speed Stability Across 1 Turns

Task   0    11.8 t/s


## Warm Run: medium prompt

# Unknown — Cache Run Analysis

**Tasks captured:** 1 (long-context >5000 tokens: 0, short-context: 1)

## Performance by Workload Type

| Metric | Long Context | Short Context | Overall |
| :--- | :---: | :---: | :---: |
| **Prompt Processing** |  N/A t/s |  8.4 t/s |  8.4 t/s |
| **Decode (Generation)** |  N/A t/s |  12.3 t/s |  12.3 t/s |

## Decode Speed Distribution

- **p50 decode:** 12.3 t/s
- **p95 decode:** 12.3 t/s

## Decode Speed Stability Across 1 Turns

Task   0    12.3 t/s


## Warm Run: small prompt

# Unknown — Cache Run Analysis

**Tasks captured:** 1 (long-context >5000 tokens: 0, short-context: 1)

## Performance by Workload Type

| Metric | Long Context | Short Context | Overall |
| :--- | :---: | :---: | :---: |
| **Prompt Processing** |  N/A t/s |  10.9 t/s |  10.9 t/s |
| **Decode (Generation)** |  N/A t/s |  12.6 t/s |  12.6 t/s |

## Decode Speed Distribution

- **p50 decode:** 12.6 t/s
- **p95 decode:** 12.6 t/s

## Decode Speed Stability Across 1 Turns

Task   0    12.6 t/s


## Cold vs Warm Speedup

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Cache | MoE hit |
|------|-----------|-----------|--------------|-------------|-------------|-------|---------|
| large | 17071.4ms | 184.4ms | 92.58x | 1.1ms | 184.4ms | ssd_warm | - |
| medium | 5902.1ms | 118.6ms | 49.76x | 1.1ms | 118.6ms | ssd_warm | - |
| small | 1798.3ms | 91.8ms | 19.59x | 1.5ms | 91.8ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint or system cache hit, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE with --moe-expert-residency)