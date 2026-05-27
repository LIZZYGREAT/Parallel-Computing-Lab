import os
import re
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "files")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

QUERY_THREADS = 4
QUERY_NPROBE = 64

ADC_STAGES = [
    "1_Coarse_Dist", "2_Coarse_Sort", "3_Compute_Residual", "4_Build_LUT",
    "5_FastScan_ADC", "6_Local_TopK_Trim", "7_Thread_Merge", "8_Global_Merge",
    "8.5_Re_Rank", "9_Build_Result",
]
SDC_STAGES = [
    "1_Coarse_Dist", "2_Coarse_Sort", "3_Compute_Residual", "4_Quantize_Query",
    "4.5_Build_LUT", "5_FastScan_SDC", "6_Local_TopK_Trim", "7_Thread_Merge",
    "8_Global_Merge", "8.5_Re_Rank", "9_Build_Result",
]
STAGE_LABELS = {
    "1_Coarse_Dist": "Coarse Dist",
    "2_Coarse_Sort": "Coarse Sort",
    "3_Compute_Residual": "Residual",
    "4_Build_LUT": "Build LUT",
    "4_Quantize_Query": "Quantize Query",
    "4.5_Build_LUT": "LUT Pack (SDC)",
    "5_FastScan_ADC": "FastScan ADC",
    "5_FastScan_SDC": "FastScan SDC",
    "6_Local_TopK_Trim": "Local TopK",
    "7_Thread_Merge": "Thread Merge",
    "8_Global_Merge": "Global Merge",
    "8.5_Re_Rank": "Exact Re-rank",
    "9_Build_Result": "Build Result",
}


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)
    return FIG_DIR


def load_tradeoff(path=None):
    path = path or os.path.join(DATA_DIR, "ivfpq_tradeoff.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["Method"].isin(["ADC", "SDC"])]
    for c in ("Threads", "NProbe"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Latency(us)"] = pd.to_numeric(df["Latency(us)"], errors="coerce")
    df["Recall@10"] = pd.to_numeric(df["Recall@10"], errors="coerce")
    df = df.dropna(subset=["Threads", "NProbe", "Latency(us)"])
    df["Threads"] = df["Threads"].astype(int)
    df["NProbe"] = df["NProbe"].astype(int)
    df = df.drop_duplicates(subset=["Method", "Threads", "NProbe"], keep="last")
    return df


def load_profiler_details(data_dir=None):
    data_dir = data_dir or DATA_DIR
    pattern = re.compile(r"profiler_detail_(ADC|SDC)_T(\d+)_P(\d+)\.csv")
    rows = []
    for fp in glob.glob(os.path.join(data_dir, "profiler_detail_*_T*_P*.csv")):
        m = pattern.search(os.path.basename(fp))
        if not m:
            continue
        method, threads, nprobe = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            df = pd.read_csv(fp, header=None, names=["Stage", "Total_us", "Avg_us"])
            for _, r in df.iterrows():
                rows.append({
                    "Method": method,
                    "Threads": threads,
                    "NProbe": nprobe,
                    "Stage": r["Stage"],
                    "Total_us": float(r["Total_us"]),
                    "Avg_us": float(r["Avg_us"]),
                })
        except Exception as e:
            print(f"[Warning] skip {fp}: {e}")
    return pd.DataFrame(rows)


def load_query_profiler(method, threads=QUERY_THREADS, nprobe=QUERY_NPROBE, data_dir=None):
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, f"profiler_detail_{method}_T{threads}_P{nprobe}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=None, names=["Stage", "Total_us", "Avg_us"])
    df["Method"] = method
    df["Threads"] = threads
    df["NProbe"] = nprobe
    return df


def load_build_profiler(path=None):
    path = path or os.path.join(DATA_DIR, "profiler_build.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=None, names=["Stage", "Total_us", "Avg_us"])
    return df
