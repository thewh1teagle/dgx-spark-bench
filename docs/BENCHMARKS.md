# Benchmarks

Methodology, measured results, and the caveats that go with them.

## What it measures

1. **Roofline** — bf16 matmul TFLOP/s (8192³) and memory bandwidth (512 MB copy,
   counted as read + write). Most training steps are bandwidth-bound, so this predicts
   much of the rest.
2. **Train steps** (300M, 1.3B) — real fwd + bwd + fused AdamW on a Llama-shaped model
   (RMSNorm, SDPA causal attention, SwiGLU), bf16, batch 1 / seq 2048. Download-free, so
   this is the tier a borrowed machine can always run.
3. **renikud** (`--renikud`) — the real architecture: `dicta-il/dictabert-large-char`
   encoder plus the three coupled classification heads, fp16 autocast + GradScaler,
   grad-clip 1.0, AdamW with discriminative LRs (2e-5 encoder / 1e-4 heads), batch 16 /
   seq 256 — matching [renikud](https://github.com/renikud/renikud)'s own training
   defaults. Random batches, so no dataset is needed and it runs identically anywhere.
   Downloads ~700 MB on first run.
4. **Extrapolated** (7B, 30B) — times a single transformer block and scales by layer
   count (+10% for embedding, head, and optimizer), so it works on cards that can't hold
   the full model. Marked `*`. Also reports estimated bf16 and QLoRA memory footprints.

Every timing is the **median** of 10 iterations after 3 warmup steps, with
`torch.cuda.synchronize()` around each. Reports `it/s`, throughput, peak VRAM, and power
draw.

Compare on **tokens/s**, not `it/s` — `it/s` is only meaningful at a fixed batch and
sequence length.

## Results

### DGX Spark (GB10) — measured 2026-08-24

`NVIDIA GB10` · 130.7 GB unified · sm_12.1 · aarch64, 20 cores · driver 580.126.09 ·
torch 2.13.0+cu130. Full run: **31 seconds**.

| tier | it/s | throughput | peak VRAM | power |
|---|---|---|---|---|
| bf16 matmul (8192³) | — | **92.5 TFLOP/s** | — | — |
| memory bandwidth | — | **219 GB/s** | — | — |
| 300M (seq 2048, b1) | 7.58 | 15,529 tok/s | 5.1 GB | 61 W |
| 1.3B (seq 2048, b1) | 2.58 | 5,284 tok/s | 12.9 GB | 67 W |
| 7B * | 0.78 | 1,601 tok/s | 13.5 GB weights | 45 W |
| 30B * | 0.17 | 347 tok/s | 65.1 GB weights | 85 W |
| **renikud** (b16, seq 256, fp16) | **2.7** | 43.2 samples/s | 8.5 GB | — |

Raw JSON: [`results/results-NVIDIA_GB10.json`](../results/results-NVIDIA_GB10.json).

Notes from the first run:

- **Bandwidth came in at 219 GB/s**, ~80% of the 273 GB/s spec — a normal
  copy-benchmark efficiency, so treat 273 as unreachable in practice.
- **renikud confirms 307M params**, and peaks at 8.5 GB. That leaves ~122 GB of the
  Spark's memory doing nothing. For this workload the Spark's one advantage is unused.
- **30B is genuinely trainable here** — 65 GB of bf16 weights fits, which no 24/32 GB
  card can do without 4-bit quantization. But at 347 tok/s it is very slow.
- **Power draw was 45–85 W** against a box rated far higher, which suggests the GPU is
  not saturated at batch 1. Larger batches would likely improve tok/s meaningfully —
  worth a sweep before drawing final conclusions.

### All GPUs — batch 1, measured 2026-08-24

All runs on torch 2.13.0 (cu130 or cu132), seq 2048 for the LM tiers, seq 256 batch 16
fp16 for renikud. RTX cards rented on vast.ai.

| | **GB10** | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---|---|---|---|
| memory | 130.7 GB unified | 25.3 GB | 25.3 GB | 33.7 GB |
| power limit | — | 420 W | 450 W | 575 W |
| bf16 matmul | 93.7 TFLOP/s | 77.5 | 162.2 | **237.0** |
| bandwidth | 221 GB/s | 840 | 913 | **1506** |
| 300M | 15,567 tok/s | 20,268 | 40,022 | **51,257** |
| 1.3B | 5,305 tok/s | 6,917 | 13,524 | **18,403** |
| 7B * | 1,618 tok/s | 1,580 | 3,282 | **4,512** |
| 30B * | 350 tok/s | 336 | 699 | **1,014** |
| **renikud** | 43.1 samp/s | 80.7 | 151.7 | **200.6** |
| power under load | 43–86 W | 182–415 W | 146–442 W | 108–511 W |

`*` extrapolated from single-block timing.

### Peak sustained throughput (batch sweep)

`it/s` is not comparable across batch sizes — an iteration at b1 and b64 are different
units of work. What matters is data through the model per second, at whatever batch each
machine runs best. This is that number.

| | 300M | 1.3B | renikud |
|---|---|---|---|
| **GB10** | 16,742 tok/s @b4 | 6,747 tok/s @b8 | 68.1 samp/s @b256 † |
| RTX 3090 | 22,404 tok/s @b4 | 6,880 tok/s @b2 | 99.0 samp/s @b64 |
| RTX 4090 | 45,135 tok/s @b4 | 14,320 tok/s @b2 | 188.0 samp/s @b64 |
| RTX 5090 | **66,977 tok/s @b8** | **21,561 tok/s @b4** | **275.9 samp/s @b64** |

† The GB10 was still gaining 4.1% per doubling when it hit the sweep's batch cap, at
70 GB of 131 GB. Its renikud figure is a floor, not a peak; every RTX card had already
OOM'd by b128. The cap has since been raised, but these numbers predate that.

Batching helps the GB10 most (+58% on renikud) because it is the only machine with the
memory to keep climbing. That narrows the renikud gap from 4.7x to **4.05x** against the
5090, and 1.9x to 1.45x against the 3090 — but does not change the order.

### What the numbers say

**On renikud, the workload that actually matters here:** the 5090 is **4.7x** the Spark
at batch 16. The 4090 is 3.5x, the 3090 1.9x.

**The Spark only ever beats the 3090**, and only on the compute-bound extrapolated
tiers (7B, 30B) where its Blackwell tensor cores out-muscle Ampere — 93.7 vs 77.5
TFLOP/s. Against the 4090 and 5090 it loses every single tier.

**Its 128 GB converts into throughput only weakly.** Large batches do help it more than
anyone else (+58% on renikud vs +23–37% for the RTX cards, because they OOM first). But
the ceiling is low: the Spark's best 1.3B result needed batch 8 and 45.3 GB — a
configuration no 24 GB card can hold — and it *still* lost to a 3090 running batch 2 in
17.5 GB. Capacity lets it run things the others cannot; it does not make it fast.

**Perf per watt is suggestive but NOT established by this data.** The `watts` field is a
single `nvidia-smi` poll taken after the timing loop, so it frequently catches an idle
gap — the same 5090 reads 108 W on one tier and 511 W on another. The Spark is certainly
the lower-power machine, but no ratio here should be quoted. Measuring it properly needs
sampling across the timed region.

**The 3090 was power-limited, not launch-limited**, sitting pinned at its 350 W cap at
the winning batch sizes. Its poor showing is weak compute, not an artifact of the
benchmark.

## Caveats

**The renikud number is a worst case.** renikud pads dynamically to the longest item in
each batch (`src/data.py`), not to 256, so real steps are often shorter than the
benchmark's fixed seq 256.

**Real training may be CPU-bound.** renikud defaults to `--dataloader-workers 0`, single
-process data loading. If the real run is CPU-bound rather than GPU-bound, this GPU
comparison overstates the difference you'd actually feel — and that's exactly where the
Spark's ARM CPU could diverge from what any GPU benchmark predicts. Check `nvidia-smi`
utilization during a real run before reading too much into it.

**Do not compare the 7B/30B rows across architectures.** They scale one block by layer
count plus a flat 10% for embedding, head, cross-entropy and the optimizer. That tail is
bandwidth-bound while the timed block is compute-bound, so a fixed 10% understates the
overhead most on the lowest-bandwidth part — it flatters the GB10 on exactly the rows
where its "fits big models" case lives. The timed block also runs no optimizer at all,
and 30B in bf16 with AdamW state needs ~260 GB, so that row describes a configuration
that cannot run on any machine here. Order-of-magnitude only.

**Batch 1 understates the discrete cards specifically.** Both machines are far from
saturated at batch 1, but the 3090 has much more bandwidth left on the table than the
Spark does. A batch-size sweep would likely widen the 3090's lead on the small tiers,
and would not change the large-tier picture much. Treat the 1.3x figures as a floor for
the 3090, not a verdict.

## Setup notes

On the Spark, `torch --index-url https://download.pytorch.org/whl/cu130` gives
2.13.0+cu130 on aarch64 and works fine on GB10 — sm_12.1 runs on the sm_120 binaries.
GB10 reports `[N/A]` for `memory.total` and `power.limit` (unified memory), so
`nvidia-smi` output is parsed defensively.
