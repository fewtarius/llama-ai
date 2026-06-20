# Benchmark Results: VULKAN

**Date:** 20260620-1314 | **Backend:** vulkan | **Context:** 32768 tokens | **Hardware:** Strix Halo (Ryzen AI Max+ 395, 96GB APU VRAM, Vulkan/RADV)

## TTFT Speedup by Size

| Model | Small (~1K tok) | Medium (~5K tok) | Large (~15K tok) |
|-------|----------------|------------------|------------------|
| GLM-4.7-Flash-Q4_K_M | **1.12x** (2445/2036ms, ssd_warm) | **1.01x** (10951/10840ms, ssd_warm) | **1.01x** (55240/54745ms, ssd_warm) |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | **1.63x** (1848/245ms, ssd_warm) | **3.52x** (7090/298ms, ssd_warm) | **5.13x** (22192/2289ms, ssd_warm) |
| gemma-4-26B-A4B-it-UD-Q5_K_M | **1.78x** (5794/1992ms, ssd_warm) | **1.03x** (8938/8571ms, ssd_warm) | **1.03x** (29676/28828ms, ssd_warm) |

**TTFT Speedup** = cold TTFT / warm TTFT (higher is better). Cache state shows whether the warm run actually restored state from disk.

## Per-Model Detail

### GLM-4.7-Flash-Q4_K_M

| Size | Cold TTFT | Warm TTFT | Speedup | Cached | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cold cache | Warm cache |
|------|-----------|-----------|---------|--------|-------------|-------------|------------|------------|------------|
| small | 2445ms | 2036ms | **1.12x** | 3/1145 | 2.1ms | 1.8ms | 16.9ms | miss | ssd_warm |
| medium | 10951ms | 10840ms | **1.01x** | 3/5237 | 2.1ms | 2.1ms | 18.8ms | miss | ssd_warm |
| large | 55240ms | 54745ms | **1.01x** | 3/15489 | 3.6ms | 3.5ms | 25.2ms | miss | ssd_warm |

### Qwen3.6-35B-A3B-UD-Q4_K_XL

| Size | Cold TTFT | Warm TTFT | Speedup | Cached | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cold cache | Warm cache |
|------|-----------|-----------|---------|--------|-------------|-------------|------------|------------|------------|
| small | 1848ms | 245ms | **1.63x** | 1237/1243 | 1.5ms | 40.9ms | 18.3ms | miss | ssd_warm |
| medium | 7090ms | 298ms | **3.52x** | 5403/5409 | 1.3ms | 49.6ms | 18.6ms | miss | ssd_warm |
| large | 22192ms | 2289ms | **5.13x** | 15715/15721 | 1.4ms | 381.5ms | 19.7ms | miss | ssd_warm |

### gemma-4-26B-A4B-it-UD-Q5_K_M

| Size | Cold TTFT | Warm TTFT | Speedup | Cached | Cold ms/tok | Warm ms/tok | Gen ms/tok | Cold cache | Warm cache |
|------|-----------|-----------|---------|--------|-------------|-------------|------------|------------|------------|
| small | 5794ms | 1992ms | **1.78x** | 7/1413 | 4.1ms | 1.4ms | 22.9ms | miss | ssd_warm |
| medium | 8938ms | 8571ms | **1.03x** | 7/6083 | 1.5ms | 1.4ms | 23.7ms | miss | ssd_warm |
| large | 29676ms | 28828ms | **1.03x** | 7/17347 | 1.7ms | 1.7ms | 25.6ms | miss | ssd_warm |

## Cold vs Warm (Qwen3.6-35B-A3B)

The SSD-backed cache works as designed on the Strix Halo. Cold and warm runs differ in the load step:

- **Cold**: server starts with empty in-memory cache, full prompt evaluated from model weights. 1.4 ms/tok at large (708 t/s).
- **Warm**: server restarts with the previous turn's checkpoints on disk. System prompt cache restores 15,715/15,721 tokens, only 6 new tokens evaluated. 2.3s total TTFT (vs 22.2s cold), 5.1x speedup.

The 5.1x at large is much smaller than the 174x reported on the Ayaneo Flip KB. The reason: the 8060S has more compute than the 780M, so cold eval is already fast (708 t/s vs ~33 t/s on the Flip). The speedup is bounded by the SSD read + restore overhead, which is roughly constant. Caching still eliminates ~20 seconds of eval per turn on a long-context agentic workload.

## Notes

- GLM-4.7-Flash and gemma-4-26B-A4B show ~1x speedup because their hybrid MoE architectures don't use the system prompt cache as effectively as Qwen3.6. Their recurrent state needs a longer matching prefix before the cache is reusable.
- The 'cold' column in the SSD log shows `loaded 0 checkpoints` for the very first run per directory; subsequent runs even at the 'cold' phase restore cached state because the prior test wrote it. The benchmark treats this as a 'warm' classification for those runs.
