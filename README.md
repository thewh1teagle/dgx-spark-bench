# dgx-spark-bench

Compare training throughput across DGX Spark (GB10) and RTX 3090 / 4090 / 5090, to
decide which is worth buying.

Roofline, real train steps at 300M and 1.3B, extrapolated 7B and 30B, and
[renikud](https://github.com/renikud/renikud)'s actual architecture. Runs in well under
a minute per GPU, and needs no dataset.

## Run

```sh
uv run src/bench.py --renikud
uv run src/compare.py results/*.json
```

On a fresh vast.ai box:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/thewh1teagle/dgx-spark-bench && cd dgx-spark-bench
uv run src/bench.py --renikud
```

## Measured so far

**renikud** (307M, the workload this is really about) — throughput at seq 256, fp16:

| GPU | memory | samples/s @b16 | vs Spark | peak (swept) |
|---|---|---|---|---|
| DGX Spark GB10 | 131 GB | 43 `████` | 1.0x | 68 @b256 † |
| RTX 3090 | 24 GB | 81 `████████` | 1.9x | 99 @b64 |
| RTX 4090 | 24 GB | 152 `███████████████` | 3.5x | 188 @b64 |
| RTX 5090 (575W) | 32 GB | 201 `████████████████████` | 4.7x | 276 @b64 |
| RTX 5090 (450W) | 32 GB | 208 `█████████████████████` | **4.8x** | — |

Everything else, batch 1, seq 2048:

| GPU | bf16 | bandwidth | 1.3B | 30B * |
|---|---|---|---|---|
| DGX Spark GB10 | 93.7 TFLOP/s | 221 GB/s | 5,305 tok/s | 350 tok/s |
| RTX 3090 | 77.5 | 840 | 6,917 | 336 |
| RTX 4090 | 162.2 | 913 | 13,524 | 699 |
| RTX 5090 (575W) | **237.0** | 1506 | **18,403** | **1,014** |
| RTX 5090 (450W) | 219.3 | **1512** | 17,508 | 918 |

† The Spark was still improving when it hit the sweep's batch cap, so 68 is a floor. At
peak the renikud gap narrows to **4.05x** (5090 575W), 2.76x (4090), 1.45x (3090). The
450W 5090 is a second card (driver 595.84); it was a quick run with no `--sweep`.

`*` The 7B/30B rows are extrapolated from one block and are **not comparable across
architectures** — the flat overhead term flatters low-bandwidth parts like the GB10. See
the docs.

The Spark beats only the 3090, and only on those extrapolated tiers. Large batches help
it more than anyone else (+58%, since the RTX cards OOM first), but the ceiling is low:
its best 1.3B run needed 45 GB and still lost to a 3090 using 17.5 GB. It is clearly the
lower-power machine, though this harness does not measure power well enough to quote a
perf/watt ratio.

Full tables, methodology, and caveats: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.
Raw JSON in [`results/`](results/).
