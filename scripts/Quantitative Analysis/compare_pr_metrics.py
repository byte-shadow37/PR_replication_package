from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== Direct input configuration (no command line needed) =====
FILE_A = "human_summary.csv"
FILE_B = "mix_human_agent.csv"
LABEL_A = "Human"   # e.g., "human"
LABEL_B = "Curated"   # e.g., "agent"
OUT_SUMMARY = "metrics_summary_human_agent.csv"
OUT_LONG = "metrics_long_concat.csv"
OUT_PLOTS = "plots"
OUT_TABLE = "metrics_comparison_table_human_curated.csv"

try:
    from scipy.stats import mannwhitneyu
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# one chart per figure, no specific colors/styles
NUMERIC_METRICS = [
    "time_to_close_hours",
    "time_to_close_days",
    "body_length",
    "commits",
    "changed_files",
    "additions",
    "deletions",
    "code_churn",
    "review_iterations",
    "number_of_reviewers",
    "total_comments",
    "reviewer_workload_hours",
]


BOOL_COLS = ["is_closed", "is_merged"]

def _iqr_bounds(series: pd.Series, k: float = 1.5):
    x = pd.Series(series).dropna()
    if x.empty:
        return np.nan, np.nan
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def _remove_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    x = pd.Series(series).dropna()
    if x.empty:
        return x
    lower, upper = _iqr_bounds(x, k)
    return x[(x >= lower) & (x <= upper)]

def _quantiles(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return {"min": np.nan, "q1": np.nan, "median": np.nan, "mean": np.nan, "q3": np.nan, "max": np.nan}
    return {
        "min": float(x.min()),
        "q1": float(x.quantile(0.25)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "q3": float(x.quantile(0.75)),
        "max": float(x.max()),
    }

def cliffs_delta(a, b):
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Efficient Cliff's delta: count pairwise comparisons using sorting
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    i = j = more = less = 0
    na, nb = len(a_sorted), len(b_sorted)
    while i < na:
        while j < nb and b_sorted[j] < a_sorted[i]:
            j += 1
        less += j
        i += 1
    i = j = 0
    while j < nb:
        while i < na and a_sorted[i] < b_sorted[j]:
            i += 1
        more += i
        j += 1
    # pairs = na*nb; delta = (more - less)/pairs
    pairs = na * nb
    if pairs == 0:
        return np.nan
    return float((more - less) / pairs)

def label_effect_size(delta_abs):
    if np.isnan(delta_abs):
        return "N/A"
    # thresholds from Romano et al. (2006): 0.147, 0.33, 0.474
    if delta_abs < 0.147:
        return "Negligible"
    if delta_abs < 0.33:
        return "Small"
    if delta_abs < 0.474:
        return "Medium"
    return "Large"

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def coerce_bool(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return df

def summarize_dataset(df: pd.DataFrame, name: str) -> pd.Series:
    s = pd.Series(dtype="float64")
    closed = df["is_closed"] if "is_closed" in df.columns else pd.Series([np.nan] * len(df))
    merged = df["is_merged"] if "is_merged" in df.columns else pd.Series([np.nan] * len(df))
    acc = (closed.fillna(False) | merged.fillna(False)).mean() if len(df) else np.nan
    s["acceptance_rate"] = acc

    for m in NUMERIC_METRICS:
        s[f"avg_{m}"] = df[m].mean() if m in df.columns else np.nan
    for m in NUMERIC_METRICS:
        s[f"median_{m}"] = df[m].median() if m in df.columns else np.nan

    s["num_prs"] = len(df)
    s.name = name
    return s

def make_boxplots(df_all: pd.DataFrame, out_dir: Path, dataset_col: str = "dataset"):
    out_dir.mkdir(parents=True, exist_ok=True)
    # generate one plot per metric
    for m in NUMERIC_METRICS:
        if m not in df_all.columns:
            continue
        if df_all[m].dropna().empty:
            continue
        groups = list(df_all.groupby(dataset_col))
        labels = []
        data = []
        for label, gdf in groups:
            cleaned = _remove_outliers_iqr(gdf[m], k=1.5).dropna().values
            if len(cleaned) > 0:
                labels.append(label)
                data.append(cleaned)
        if sum(len(arr) for arr in data) == 0:
            continue
        plt.figure()
        bp = plt.boxplot(
            data,
            labels=labels,
            showfliers=False,          # HIDE outliers to avoid extreme scaling
            showmeans=True,            # still show the mean explicitly
            meanline=False,            # mean as a marker (not a line)
            meanprops={"marker": "x", "markersize": 8, "markeredgewidth": 1.5, "markerfacecolor": "tab:orange", "markeredgecolor": "tab:orange"},
            medianprops={"color": "green", "linewidth": 1.5},  # distinguish median from mean
        )
        # Legend with proxy artists so the meaning is clear
        handles = [
            Line2D([], [], color="green", linestyle="-", label="Median"),
            Line2D([], [], marker="x", color="tab:orange", linestyle="None", label="Mean"),
        ]
        plt.legend(handles=handles, loc="best", frameon=False)

        plt.ylabel(m)
        plt.title(f"Distribution of {m} by dataset (IQR-cleaned, k=1.5)")
        fig_path = out_dir / f"box_{m}.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

def load_and_prepare(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset"] = dataset_name
    df = coerce_bool(df, BOOL_COLS)
    df = coerce_numeric(df, NUMERIC_METRICS)
    return df


def build_comparison_table(df_all: pd.DataFrame, label_a: str, label_b: str, dataset_col: str = "dataset") -> pd.DataFrame:
    rows = []
    g = {k: v for k, v in df_all.groupby(dataset_col)}
    A = g.get(label_a, pd.DataFrame())
    B = g.get(label_b, pd.DataFrame())

    for m in NUMERIC_METRICS:
        if m not in df_all.columns:
            continue
        a_raw = A[m] if m in A.columns else pd.Series(dtype=float)
        b_raw = B[m] if m in B.columns else pd.Series(dtype=float)

        # Raw summaries for min/Q1/Q3/Max
        qa = _quantiles(a_raw)
        qb = _quantiles(b_raw)

        # Cleaned summaries for mean/median only (remove outliers via IQR 1.5x)
        a_clean = _remove_outliers_iqr(a_raw, k=1.5)
        b_clean = _remove_outliers_iqr(b_raw, k=1.5)
        qa["mean"] = float(a_clean.mean()) if a_clean.size > 0 else np.nan
        qa["median"] = float(a_clean.median()) if a_clean.size > 0 else np.nan
        qb["mean"] = float(b_clean.mean()) if b_clean.size > 0 else np.nan
        qb["median"] = float(b_clean.median()) if b_clean.size > 0 else np.nan

        # Significance tests/effect size remain on RAW data unless specified otherwise
        if _HAS_SCIPY and a_raw.dropna().size > 0 and b_raw.dropna().size > 0:
            try:
                stat, p = mannwhitneyu(a_raw.dropna(), b_raw.dropna(), alternative="two-sided")
            except Exception:
                p = np.nan
        else:
            p = np.nan
        delta = cliffs_delta(a_raw, b_raw)
        eff_label = label_effect_size(abs(delta))

        rows.append({
            "Metric": m,
            f"{label_a} min": qa["min"],
            f"{label_a} Q1": qa["q1"],
            f"{label_a} Median": qa["median"],  # cleaned
            f"{label_a} Mean": qa["mean"],      # cleaned
            f"{label_a} Q3": qa["q3"],
            f"{label_a} Max": qa["max"],
            f"{label_b} min": qb["min"],
            f"{label_b} Q1": qb["q1"],
            f"{label_b} Median": qb["median"],  # cleaned
            f"{label_b} Mean": qb["mean"],      # cleaned
            f"{label_b} Q3": qb["q3"],
            f"{label_b} Max": qb["max"],
            "P-value": p,
            "Effect size (delta)": delta,
            "Effect label": eff_label,
        })

    return pd.DataFrame(rows)

def main():
    pA = Path(FILE_A)
    pB = Path(FILE_B)
    label_a = LABEL_A or pA.name
    label_b = LABEL_B or pB.name

    df_a = load_and_prepare(pA, label_a)
    df_b = load_and_prepare(pB, label_b)

    df_all = pd.concat([df_a, df_b], ignore_index=True)

    sum_a = summarize_dataset(df_a, label_a)
    sum_b = summarize_dataset(df_b, label_b)
    summary_df = pd.DataFrame([sum_a, sum_b])

    out_summary = Path(OUT_SUMMARY)
    out_long = Path(OUT_LONG)
    out_plots = Path(OUT_PLOTS)

    summary_df.to_csv(out_summary, index=True)
    df_all.to_csv(out_long, index=False)

    make_boxplots(df_all, out_plots)

    comp_df = build_comparison_table(df_all, label_a, label_b)
    comp_out = Path(OUT_TABLE)
    comp_df.to_csv(comp_out, index=False)
    print(f"[OK] Wrote summary: {out_summary.resolve()}")
    print(f"[OK] Wrote long form: {out_long.resolve()}")
    print(f"[OK] Plots in: {out_plots.resolve()}")
    print(f"[OK] Wrote comparison table: {comp_out.resolve()}")

if __name__ == "__main__":
    main()
