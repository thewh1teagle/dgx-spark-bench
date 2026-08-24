"""Merge results-*.json into one markdown table.  uv run src/compare.py results-*.json"""

import json
import sys


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    sizes = list(dict.fromkeys(k for r in runs for k in r["models"]))

    has_renikud = any(r.get("renikud") for r in runs)
    hdr = ["GPU", "VRAM", "TFLOP/s", "GB/s"] + [f"{s} tok/s" for s in sizes]
    if has_renikud:
        hdr.append("renikud samp/s")
    rows = []
    for r in runs:
        d = r["device"]
        row = [d["gpu"], f"{d['vram_gb']:.0f}G",
               str(r["roofline"]["matmul_tflops"]), str(r["roofline"]["bandwidth_gbs"])]
        for s in sizes:
            m = r["models"].get(s)
            if not m:
                row.append("-")
            elif m.get("oom"):
                row.append("OOM")
            else:
                row.append(f"{m['tokens_s']:,}" + ("" if m["measured"] else "*"))
        if has_renikud:
            k = r.get("renikud")
            row.append("-" if not k else "OOM" if k.get("oom") else f"{k['samples_s']}")
        rows.append(row)

    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
    print("\n* = extrapolated: one block x layers +10%. No optimizer in the timed step,")
    print("    and the +10% flatters low-bandwidth parts. NOT cross-architecture comparable.")
    print("throughput is per second of data through the model, not iterations")


def peak_table(paths):
    """Second table: best sustained throughput found by --sweep, and the batch that got it."""
    runs = [(p, json.load(open(p))) for p in paths]
    runs = [(p, r) for p, r in runs if r.get("sweep")]
    if not runs:
        return
    sizes = list(dict.fromkeys(k for _, r in runs for k in r["sweep"]))

    print("\n\nPeak sustained throughput (best point in the batch sweep)\n")
    hdr = ["GPU"] + [f"{s}" for s in sizes]
    rows = []
    for _, r in runs:
        row = [r["device"]["gpu"]]
        for size in sizes:
            key0 = "samples_s" if size == "renikud" else "tokens_s"
            pts = [p for p in r["sweep"].get(size, []) if key0 in p]
            if not pts:
                row.append("-")
                continue
            key = key0
            best = max(pts, key=lambda p: p[key])
            unit = "samp/s" if size == "renikud" else "tok/s"
            row.append(f"{best[key]:,} {unit} @b{best['batch']}")
        rows.append(row)

    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")


if __name__ == "__main__":
    paths = sys.argv[1:] or sys.exit("usage: compare.py results/*.json")
    main([p for p in paths if "sweep" not in p] or paths)
    peak_table(paths)
