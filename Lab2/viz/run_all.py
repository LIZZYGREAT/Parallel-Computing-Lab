#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from data_loader import load_tradeoff, load_profiler_details, load_build_profiler, ensure_fig_dir
from plot_speedup import plot_speedup
from plot_stage_breakdown import plot_stage_breakdown
from plot_adc_sdc import plot_latency_recall, plot_adc_sdc_ratio, plot_stage_compare


def plot_build_breakdown(out_dir):
    import matplotlib.pyplot as plt
    df = load_build_profiler()
    if df.empty:
        return
    plt.rcParams["axes.unicode_minus"] = False
    labels = [s.replace("Build_", "") for s in df["Stage"]]
    vals = df["Total_us"].values / 1e6
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.barh(labels, vals, color=plt.cm.Paired(range(len(labels))))
    ax.set_xlabel("Time (s)")
    ax.set_title("Index Build Stage Time")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "build_breakdown.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"[OK] {p}")


def main():
    ensure_fig_dir()
    tradeoff = load_tradeoff()
    profiler = load_profiler_details()
    print(f"[Data] tradeoff={len(tradeoff)} rows, profiler={len(profiler)} rows")
    plot_speedup(tradeoff)
    plot_latency_recall(tradeoff)
    plot_adc_sdc_ratio(tradeoff)
    if not profiler.empty:
        for t in sorted(profiler["Threads"].unique()):
            plot_stage_breakdown(profiler, threads=int(t))
    plot_stage_compare(profiler, threads=4, nprobe=32)
    plot_stage_compare(profiler, threads=8, nprobe=64)
    plot_build_breakdown(ensure_fig_dir())
    print(f"\n[Done] figures saved to: figures/")


if __name__ == "__main__":
    main()
