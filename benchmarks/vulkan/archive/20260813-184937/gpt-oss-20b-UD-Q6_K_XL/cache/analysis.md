# gpt-oss-20b-UD-Q6_K_XL — Cache Test (vulkan)

## Warm Run: large prompt

# Unknown — Cache Run Analysis

**Tasks captured:** 1 (long-context >5000 tokens: 0, short-context: 1)

## Performance by Workload Type

| Metric | Long Context | Short Context | Overall |
| :--- | :---: | :---: | :---: |
| **Prompt Processing** |  N/A t/s |  9.1 t/s |  9.1 t/s |
| **Decode (Generation)** |  N/A t/s |  21.1 t/s |  21.1 t/s |

## Decode Speed Distribution

- **p50 decode:** 21.1 t/s
- **p95 decode:** 21.1 t/s

## Decode Speed Stability Across 1 Turns

Task   0    21.1 t/s


## Warm Run: medium prompt

# Unknown — Cache Run Analysis

**Tasks captured:** 1 (long-context >5000 tokens: 0, short-context: 1)

## Performance by Workload Type

| Metric | Long Context | Short Context | Overall |
| :--- | :---: | :---: | :---: |
| **Prompt Processing** |  N/A t/s |  14.3 t/s |  14.3 t/s |
| **Decode (Generation)** |  N/A t/s |  21.8 t/s |  21.8 t/s |

## Decode Speed Distribution

- **p50 decode:** 21.8 t/s
- **p95 decode:** 21.8 t/s

## Decode Speed Stability Across 1 Turns

Task   0    21.8 t/s


## Warm Run: small prompt

# Unknown — Cache Run Analysis

**Tasks captured:** 1 (long-context >5000 tokens: 0, short-context: 1)

## Performance by Workload Type

| Metric | Long Context | Short Context | Overall |
| :--- | :---: | :---: | :---: |
| **Prompt Processing** |  N/A t/s |  18.1 t/s |  18.1 t/s |
| **Decode (Generation)** |  N/A t/s |  22.9 t/s |  22.9 t/s |

## Decode Speed Distribution

- **p50 decode:** 22.9 t/s
- **p95 decode:** 22.9 t/s

## Decode Speed Stability Across 1 Turns

Task   0    22.9 t/s


## Cold vs Warm Speedup

| Size | Cold TTFT | Warm TTFT | TTFT Speedup | Cold ms/tok | Warm ms/tok | Cache | MoE hit |
|------|-----------|-----------|--------------|-------------|-------------|-------|---------|
| large | 10234.5ms | 110.2ms | 2.61x | 0.7ms | 110.2ms | ssd_warm | - |
| medium | 3404.4ms | 69.8ms | 1.54x | 0.6ms | 69.8ms | ssd_warm | - |
| small | 997.8ms | 55.2ms | 1.19x | 0.8ms | 55.2ms | ssd_warm | - |

**Cache states:** `ssd_cold` = restored from SSD after restart, `ssd_warm` = in-memory checkpoint or system cache hit, `miss` = no cache hit
**TTFT** = Time To First Token (server-side prompt eval)
**TTFT Speedup** = cold TTFT / warm TTFT
**MoE hit** = warm-run MoE expert residency hit rate (only for MoE with --moe-expert-residency)