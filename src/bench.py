"""Lean GPU benchmark: roofline + transformer train-step throughput.

Core tiers are torch-only and need no dataset. Writes results/results-<gpu>.json.
    uv run src/bench.py --renikud
"""

import argparse
import json
import os
import platform
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (d_model, n_layers, n_heads, d_ffn)
CONFIGS = {
    "124m": (768, 12, 12, 2048),
    "300m": (1024, 24, 16, 2816),  # ~renikud scale
    "1.3b": (2048, 24, 16, 5632),
    "7b": (4096, 32, 32, 11008),
    "30b": (6656, 60, 52, 17920),
}
# Measured directly up to this size; larger configs are extrapolated from one block.
MEASURE_DIRECT = {"124m", "300m", "1.3b"}


class Block(nn.Module):
    def __init__(self, d, h, ffn):
        super().__init__()
        self.h = h
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        g, u = self.up(self.n2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


class Model(nn.Module):
    def __init__(self, d, layers, h, ffn, vocab=32000):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, h, ffn) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def gpu_info():
    p = torch.cuda.get_device_properties(0)
    info = {
        "gpu": p.name,
        "vram_gb": round(p.total_memory / 1e9, 1),
        "sm": f"{p.major}.{p.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch": platform.machine(),
    }
    # GB10 reports [N/A] for several of these (unified memory), so parse defensively.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0].split(", ")
        info["power_limit_w"] = _num(out[0])
        info["driver"] = out[1].strip()
    except Exception:
        pass
    return info


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def power_draw():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return _num(out.splitlines()[0])
    except Exception:
        return None


def timed(fn, warmup=3, iters=10):
    """Return median seconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def roofline(dtype=torch.bfloat16):
    """Compute peak (TFLOP/s) and memory bandwidth (GB/s)."""
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    s = timed(lambda: torch.mm(a, b))
    tflops = 2 * n**3 / s / 1e12

    x = torch.empty(2**28, device="cuda", dtype=dtype)  # 512 MB
    y = torch.empty_like(x)
    s = timed(lambda: y.copy_(x))
    gbs = 2 * x.numel() * x.element_size() / s / 1e9  # read + write

    del a, b, x, y
    torch.cuda.empty_cache()
    return {"matmul_tflops": round(tflops, 1), "bandwidth_gbs": round(gbs)}


def train_step_bench(cfg, batch, seq, iters=10):
    """Measure a real fwd+bwd+optimizer step. Returns None on OOM."""
    d, layers, h, ffn = CONFIGS[cfg]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = Model(d, layers, h, ffn).to("cuda", torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        ids = torch.randint(0, 32000, (batch, seq), device="cuda")

        def step():
            loss = F.cross_entropy(
                model(ids).view(-1, 32000).float(), ids.view(-1)
            )
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        s = timed(step, warmup=3, iters=iters)
        watts = power_draw()
        peak = torch.cuda.max_memory_allocated() / 1e9
        params = sum(p.numel() for p in model.parameters())
        del model, opt, ids
        torch.cuda.empty_cache()
        return {
            "params_b": round(params / 1e9, 2),
            "batch": batch, "seq": seq,
            "it_s": round(1 / s, 3),
            "tokens_s": round(batch * seq / s),
            "peak_vram_gb": round(peak, 1),
            "watts": watts,
            "measured": True,
        }
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def extrapolate(cfg, batch, seq):
    """Time one block, scale by layer count. For models too big to instantiate."""
    d, layers, h, ffn = CONFIGS[cfg]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    block = Block(d, h, ffn).to("cuda", torch.bfloat16)
    x = torch.randn(batch, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step():
        block(x).sum().backward()
        block.zero_grad(set_to_none=True)
        x.grad = None

    s = timed(step, warmup=3, iters=8)
    watts = power_draw()
    # +10% stands in for embedding, the 32k-vocab head, fp32 cross-entropy and the
    # AdamW step. Those are bandwidth-bound while the timed block is compute-bound, so
    # this FLATTERS low-bandwidth parts (GB10) relative to high-bandwidth ones. The
    # timed block also runs no optimizer at all. Order-of-magnitude only; do not use
    # these rows for cross-architecture comparison.
    step_s = s * layers * 1.1
    params = layers * sum(p.numel() for p in block.parameters()) + 2 * 32000 * d
    del block, x
    torch.cuda.empty_cache()
    return {
        "params_b": round(params / 1e9, 2),
        "batch": batch, "seq": seq,
        "it_s": round(1 / step_s, 4),
        "tokens_s": round(batch * seq / step_s),
        "bf16_weights_gb": round(params * 2 / 1e9, 1),
        "qlora_est_gb": round(params * 0.55 / 1e9, 1),  # nf4 + lora + optimizer
        "watts": watts,
        "measured": False,
    }


def is_unified_memory():
    """True when GPU memory is carved from system RAM (GB10 / Jetson-class).

    Matters because an oversized batch there is reaped by the kernel OOM killer
    rather than raising torch.OutOfMemoryError.
    """
    try:
        p = torch.cuda.get_device_properties(0)
        return platform.machine() == "aarch64" and p.total_memory > 60e9
    except Exception:
        return False


def mem_available_gb():
    """Free system RAM. Only meaningful on unified-memory boxes."""
    if not is_unified_memory():
        return float("inf")  # discrete GPU: host RAM is irrelevant to VRAM headroom
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("inf")


# On unified memory the next doubling can get the process SIGKILLed (kernel OOM killer,
# or an oomguard-style daemon) before torch can raise a catchable OutOfMemoryError -- so
# there is no exception to catch and no partial result. Stop while the NEXT rung would
# still fit comfortably, rather than measuring only what already fit.
MEM_FLOOR_GB = 24
SAFE_FRACTION = 0.65


def next_rung_unsafe(peak_gb):
    """True if doubling from peak_gb would risk a SIGKILL on a unified-memory box."""
    if not is_unified_memory():
        return False  # discrete GPU raises a catchable OOM instead
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return peak_gb * 2 > SAFE_FRACTION * total_gb


def sweep(cfg, seq, max_batch=2048):
    """Double batch size until OOM or throughput stops improving. Returns all points."""
    points, best, b, stale = [], 0.0, 1, 0
    while b <= max_batch:
        res = train_step_bench(cfg, b, seq, iters=5)
        if res is None:
            points.append({"batch": b, "oom": True})
            break
        points.append(res)
        # Require TWO consecutive non-improving doublings before stopping. A single
        # noisy point (wave quantization, thermal dip) must not end the sweep -- that
        # bug once reported a 19% low outlier as the peak.
        if res["tokens_s"] < best * 1.03:
            stale += 1
            if stale >= 2:
                break
        else:
            stale = 0
        if next_rung_unsafe(res["peak_vram_gb"]) or mem_available_gb() < MEM_FLOOR_GB:
            points.append({"batch": b * 2, "skipped": "memory headroom"})
            break
        best = max(best, res["tokens_s"])
        b *= 2
    if b > max_batch:
        points.append({"batch": b, "capped": max_batch})
    return points


def sweep_renikud(seq, max_batch=2048):
    from renikud_bench import bench as rb, load_model

    model = load_model()  # load once, reuse across batch sizes
    points, best, b, stale = [], 0.0, 16, 0
    while b <= max_batch:
        res = rb(b, seq, iters=5, model=model)
        if res is None:
            points.append({"batch": b, "oom": True})
            break
        points.append(res)
        if res["samples_s"] < best * 1.03:
            stale += 1
            if stale >= 2:
                break
        else:
            stale = 0
        if next_rung_unsafe(res["peak_vram_gb"]) or mem_available_gb() < MEM_FLOOR_GB:
            points.append({"batch": b * 2, "skipped": "memory headroom"})
            break
        best = max(best, res["samples_s"])
        b *= 2
    if b > max_batch:
        points.append({"batch": b, "capped": max_batch})
    del model
    torch.cuda.empty_cache()
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sizes", default="300m,1.3b,7b,30b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--renikud", action="store_true",
                    help="also bench renikud's real architecture (downloads ~700MB encoder)")
    ap.add_argument("--renikud-batch", type=int, default=16)
    ap.add_argument("--renikud-seq", type=int, default=256)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep batch size upward until OOM or the compute roof")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    info = gpu_info()
    print(f"\n{info['gpu']}  |  {info['vram_gb']} GB  |  torch {info['torch']}  |  {info['arch']}\n")

    results = {"device": info, "seq": args.seq, "roofline": None, "models": {}}

    print("roofline...", flush=True)
    results["roofline"] = roofline()
    r = results["roofline"]
    print(f"  bf16 matmul  {r['matmul_tflops']:>8} TFLOP/s")
    print(f"  bandwidth    {r['bandwidth_gbs']:>8} GB/s\n")

    results["sweep"] = {} if args.sweep else None

    for cfg in args.sizes.split(","):
        cfg = cfg.strip()
        if cfg not in CONFIGS:
            continue
        print(f"{cfg}...", end=" ", flush=True)
        if cfg in MEASURE_DIRECT:
            if args.sweep:
                pts = sweep(cfg, args.seq)
                results["sweep"][cfg] = pts
                ok = [p for p in pts if not p.get("oom")]
                res = max(ok, key=lambda p: p["tokens_s"]) if ok else None
                if res:
                    trail = " ".join(
                        f"b{p['batch']}:OOM" if p.get("oom") else f"b{p['batch']}:{p['tokens_s']//1000}k"
                        for p in pts
                    )
                    print(f"best b{res['batch']}  ", end="")
            else:
                res = train_step_bench(cfg, args.batch, args.seq)
                trail = ""
            if res is None:
                print("OOM")
                results["models"][cfg] = {"oom": True}
                continue
        else:
            res = extrapolate(cfg, args.batch, args.seq)
            trail = ""
        results["models"][cfg] = res
        tag = "" if res["measured"] else "  (extrapolated)"
        print(f"{res['it_s']:>8} it/s   {res['tokens_s']:>7} tok/s   "
              f"peak {res.get('peak_vram_gb', res.get('bf16_weights_gb'))} GB{tag}"
              + (f"   [{trail}]" if trail else ""))

    if args.renikud:
        print("renikud...", end=" ", flush=True)
        try:
            from renikud_bench import bench as renikud_bench
            if args.sweep:
                pts = sweep_renikud(args.renikud_seq)
                results["sweep"]["renikud"] = pts
                ok = [p for p in pts if not p.get("oom")]
                res = max(ok, key=lambda p: p["samples_s"]) if ok else None
                if res:
                    print(f"best b{res['batch']}  ", end="")
            else:
                res = renikud_bench(args.renikud_batch, args.renikud_seq)
            if res is None:
                print("OOM")
                results["renikud"] = {"oom": True}
            else:
                results["renikud"] = res
                print(f"{res['it_s']:>8} it/s   {res['samples_s']:>7} samp/s   "
                      f"peak {res['peak_vram_gb']} GB   ({res['params_m']}M params)")
        except ImportError:
            print("skipped (pip install transformers)")

    print()
    out = args.out or f"results/results-{info['gpu'].replace(' ', '_')}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
