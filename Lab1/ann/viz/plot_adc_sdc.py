import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import load_tradeoff, load_profiler_details, ensure_fig_dir, ADC_STAGES, SDC_STAGES

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS"]


def plot_latency_recall(tradeoff=None, out_dir=None):
    df = tradeoff if tradeoff is not None else load_tradeoff()
    out_dir = out_dir or ensure_fig_dir()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    markers = {"ADC": "o", "SDC": "s"}
    colors = {"ADC": "#1f77b4", "SDC": "#ff7f0e"}
    for method in ("ADC", "SDC"):
        sub = df[df["Method"] == method]
        for j, t in enumerate(sorted(sub["Threads"].unique())):
            s = sub[sub["Threads"] == t].sort_values("Recall@10")
            axes[0].plot(
                s["Recall@10"], s["Latency(us)"], f"{markers[method]}-",
                color=colors[method], alpha=0.45 + 0.12 * j, linewidth=1.5,
                label=f"{method} T={t}",
            )
    axes[0].set_xlabel("Recall@10")
    axes[0].set_ylabel("Latency (us)")
    axes[0].set_title("Recall-Latency 权衡曲线")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=7, ncol=2)

    t_fix = 4
    nprobes = sorted(df["NProbe"].unique())
    x = np.arange(len(nprobes))
    w = 0.35
    for i, method in enumerate(("ADC", "SDC")):
        sub = df[(df["Method"] == method) & (df["Threads"] == t_fix)].set_index("NProbe")["Latency(us)"]
        lat = [sub.get(p, np.nan) for p in nprobes]
        axes[1].bar(x + (i - 0.5) * w, lat, width=w, label=method, color=colors[method], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(p) for p in nprobes])
    axes[1].set_xlabel("NProbe")
    axes[1].set_ylabel("Latency (us)")
    axes[1].set_title(f"ADC vs SDC 延迟对比 (T={t_fix})")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(out_dir, "adc_sdc_tradeoff.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_adc_sdc_ratio(tradeoff=None, out_dir=None):
    df = tradeoff if tradeoff is not None else load_tradeoff()
    out_dir = out_dir or ensure_fig_dir()
    if df.empty:
        return
    adc = df[df["Method"] == "ADC"].set_index(["Threads", "NProbe"])["Latency(us)"]
    sdc = df[df["Method"] == "SDC"].set_index(["Threads", "NProbe"])["Latency(us)"]
    common = adc.index.intersection(sdc.index)
    ratio = _latency_ratio(adc, sdc, common)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    threads = sorted({t for t, _ in common})
    nprobes = sorted({p for _, p in common})
    for t in threads:
        ys = [ratio.get((t, p), np.nan) for p in nprobes]
        ax.plot(nprobes, ys, "o-", linewidth=2, label=f"T={t}")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("NProbe")
    ax.set_ylabel("SDC延迟 / ADC延迟")
    ax.set_title("ADC vs SDC 相对延迟 (<1 表示 SDC 更快)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "adc_sdc_latency_ratio.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def _latency_ratio(adc, sdc, common):
    d = {}
    for k in common:
        a, s = adc.loc[k], sdc.loc[k]
        if a > 0:
            d[k] = s / a
    return d


def plot_stage_compare(profiler=None, threads=4, nprobe=32, out_dir=None):
    prof = profiler if profiler is not None else load_profiler_details()
    out_dir = out_dir or ensure_fig_dir()
    if prof.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    groups = []
    vals = []
    for method, stages, scan in [("ADC", ADC_STAGES, "5_ADC_Scan"), ("SDC", SDC_STAGES, "5_SDC_Scan")]:
        sub = prof[(prof["Method"] == method) & (prof["Threads"] == threads) & (prof["NProbe"] == nprobe)]
        if sub.empty:
            continue
        row = sub.set_index("Stage")["Avg_us"]
        lut_q = float(row.get(stages[3], 0))
        scan_v = float(row.get(scan, 0))
        coarse = float(row.get("1_Coarse_Dist", 0)) + float(row.get("2_Coarse_Sort", 0))
        other = float(row.sum()) - lut_q - scan_v - coarse
        groups.extend([f"{method}\n粗排", f"{method}\n量化/LUT", f"{method}\n扫描", f"{method}\n其他"])
        vals.extend([coarse, lut_q, scan_v, other])

    x = np.arange(len(vals))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#bab0ac"] * 2
    ax.bar(x, vals, color=colors[: len(vals)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("平均耗时 (us/query)")
    ax.set_title(f"ADC vs SDC 主要阶段耗时 (T={threads}, nprobe={nprobe})")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(out_dir, f"adc_sdc_stages_T{threads}_P{nprobe}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    plot_latency_recall()
    plot_adc_sdc_ratio()
    plot_stage_compare()
