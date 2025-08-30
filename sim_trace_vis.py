#!/usr/bin/env python3
import argparse, glob, re
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Ev = namedtuple("Ev", "rank kind value")

def read_one(path, rank):
    evs = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            m = re.match(r"(allreduce|compute)\s+([0-9.]+)", ln)
            if not m:
                raise ValueError(f"{path}: bad line '{ln}'")
            k, v = m.groups()
            v = int(v) if k == "allreduce" else float(v)
            evs.append(Ev(rank, k, v))
    return evs

def load_all():
    files = sorted(glob.glob("rank_*.trace"))
    if not files:
        raise RuntimeError("no rank_*.trace files found")
    all_evs = []
    for r, f in enumerate(files):
        all_evs.extend(read_one(f, r))
    return all_evs, len(files)

def build_iters(evs, P):
    by_rank = defaultdict(list)
    for e in evs:
        by_rank[e.rank].append(e)
    steps = {len(v) for v in by_rank.values()}
    if len(steps) != 1:
        raise RuntimeError("rank traces have unequal length")
    n_iter = steps.pop() // 2
    rows = []
    for i in range(n_iter):
        comp = bytes_ = 0.0
        for seq in by_rank.values():
            a, c = seq[2*i], seq[2*i+1]
            if a.kind != "allreduce":
                a, c = c, a
            bytes_ += a.value
            comp += c.value
        rows.append(dict(iter=i,
                         compute_s=comp / P,
                         bytes=bytes_ / P))
    return pd.DataFrame(rows)

def comm_time(bytes_, P, bw, latency, algo):
    β = 1.0 / bw
    if algo == "flat":
        fac = 1.0
    elif algo == "ring":
        fac = (P - 1) / P
    elif algo == "tree":
        fac = 2.0
    else:
        raise ValueError("algo must be flat, ring or tree")
    return latency + β * bytes_ * fac

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth", "-b", type=float, default=10e9)
    ap.add_argument("--latency", "-l", type=float, default=0.0)
    ap.add_argument("--algo", "-a", choices=["flat", "ring", "tree"], default="flat")
    ap.add_argument("--savefig", default="trace_timeline.png")
    ap.add_argument("--ratio", "-r", type=float, default=1.0)
    args = ap.parse_args()

    latency_s = args.latency * 1e-6  # Convert µs → s
    eff_bw = args.bandwidth * args.ratio
    evs, P = load_all()
    df = build_iters(evs, P)
    df["comm_s"] = df["bytes"].apply(lambda b: comm_time(b, P, eff_bw, latency_s, args.algo))
    df["step_s"] = df["compute_s"] + df["comm_s"]

    # Cumulative timeline for stacked compute → comm per iteration
    df["start"] = df["step_s"].cumsum().shift(fill_value=0)
    df["compute_start"] = df["start"]
    df["comm_start"] = df["compute_start"] + df["compute_s"]
    df["comm_s_baseline"] = df["bytes"].apply(
        lambda b: comm_time(b, P, args.bandwidth, latency_s, args.algo)
    )
    
    T_baseline   = df["compute_s"].sum() + df["comm_s_baseline"].sum()
    T_compressed = df["compute_s"].sum() + df["comm_s"].sum()
    speedup = T_baseline / T_compressed
    # -------- Summary ----------
    print("\n===  GLOBAL SUMMARY  ===")
    print(f"ranks           : {P}")
    print(f"iterations      : {len(df)}")
    print(f"bandwidth       : {args.bandwidth / 1e9:.2f} Gb/s")
    print(f"ratio           : {args.ratio}x")
    print(f"latency α       : {args.latency:.1f} µs")
    print(f"algorithm fac   : {args.algo}")
    tot = df["compute_s"].sum() + df["comm_s"].sum()
    print(f"total compute   : {df['compute_s'].sum():.2f} s")
    print(f"total comm      : {df['comm_s'].sum():.2f} s")
    print(f"comm/compute      : {100 * df['comm_s'].sum() / df['compute_s'].sum():.2f} %")
    print(f"speedup          : {speedup:.2f} ×")
    # -------- Plot timeline --------
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.barh(0, width=df["compute_s"], left=df["compute_start"],
            height=0.5, label="compute", color="steelblue")
    ax.barh(0, width=df["comm_s"], left=df["comm_start"],
            height=0.5, label="communication", color="firebrick")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_yticks([])
    ax.set_title("Execution timeline for a Node")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(args.savefig, dpi=150)
    print(f"\nPlot written to {args.savefig}")

if __name__ == "__main__":
    main()

