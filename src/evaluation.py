"""
Evaluation Metrics for Accent Classification

Includes Tarui preservation analysis and method comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]


def compute_tarui_preservation(
    df: pd.DataFrame,
    method_suffix: str,
    min_rate: float = 0.15,
    max_rate: float = 0.80,
    ratio_lo: float = 0.7,
    ratio_hi: float = 1.3,
) -> Dict:
    """
    Evaluate whether a method preserves the Tarui accent category.

    Tarui is a transitional accent variety between Kansai and Tokyo-type
    accents. Non-temporal distance metrics (Mean F0, Histogram EMD) tend
    to collapse it into Kansai, while over-sensitive metrics may over-assign
    subjects to Tarui. A well-calibrated method preserves Tarui at a
    moderate rate with balanced Tarui/Kansai distance ratios.

    Conditions for 'preserved':
      (A) Tarui classification rate ≥ min_rate (not collapsed)
      (B) Tarui classification rate ≤ max_rate (not over-assigned)
      (C) Mean Tarui/Kansai distance ratio within [ratio_lo, ratio_hi]

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with columns '{accent}{method_suffix}' for each
        accent type (e.g., 'tokyo_dtw', 'kansai_dtw', etc.)
    method_suffix : str
        Column suffix identifying the method (e.g., '_dtw', '_meanf0')
    min_rate : float
        Minimum acceptable Tarui classification rate
    max_rate : float
        Maximum acceptable Tarui classification rate
    ratio_lo : float
        Lower bound of acceptable Tarui/Kansai distance ratio
    ratio_hi : float
        Upper bound of acceptable Tarui/Kansai distance ratio

    Returns
    -------
    dict
        Evaluation results with keys:
        - method: method identifier
        - tarui_classification_rate: fraction classified as Tarui
        - mean_tarui_kansai_ratio: mean distance ratio
        - tarui_preserved: bool (all conditions met)
        - tarui_overassigned: bool (rate > max_rate)
        - tarui_collapsed: bool (rate < min_rate)
    """
    def get_nearest(row):
        dists = {a: row.get(f"{a}{method_suffix}", np.inf) for a in ACCENT_TYPES}
        finite = {k: v for k, v in dists.items() if np.isfinite(v)}
        return min(finite, key=finite.get) if finite else "unknown"

    tmp = df.copy()
    tmp["nearest"] = tmp.apply(get_nearest, axis=1)
    tarui_rate = float((tmp["nearest"] == "tarui").mean())

    ratios = []
    for _, row in tmp.iterrows():
        t = row.get(f"tarui{method_suffix}", np.inf)
        k = row.get(f"kansai{method_suffix}", np.inf)
        if np.isfinite(t) and np.isfinite(k) and k > 0:
            ratios.append(t / k)
    mean_ratio = float(np.mean(ratios)) if ratios else np.nan

    preserved = min_rate <= tarui_rate <= max_rate
    if np.isfinite(mean_ratio):
        preserved = preserved and (ratio_lo <= mean_ratio <= ratio_hi)

    return {
        "method": method_suffix.strip("_"),
        "tarui_classification_rate": tarui_rate,
        "mean_tarui_kansai_ratio": mean_ratio,
        "tarui_preserved": bool(preserved),
        "tarui_overassigned": tarui_rate > max_rate,
        "tarui_collapsed": tarui_rate < min_rate,
    }


def generate_comparison_table(
    df_dtw: pd.DataFrame,
    df_baseline: pd.DataFrame,
    out_path: str,
) -> pd.DataFrame:
    """
    Generate a comparison table of Tarui preservation across all methods.

    Parameters
    ----------
    df_dtw : pd.DataFrame
        DTW classification results (columns: '{accent}_distance')
    df_baseline : pd.DataFrame
        Baseline results (columns: '{accent}_meanf0', '{accent}_histemd',
        '{accent}_wav2vec')
    out_path : str
        Path to save CSV output

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    results = []

    # Proposed method (DTW)
    if df_dtw is not None and not df_dtw.empty:
        df_tmp = df_dtw.copy()
        for a in ACCENT_TYPES:
            if f"{a}_distance" in df_tmp.columns:
                df_tmp[f"{a}_dtw"] = df_tmp[f"{a}_distance"]
        results.append(compute_tarui_preservation(df_tmp, "_dtw"))

    # Baselines
    if df_baseline is not None and not df_baseline.empty:
        for suffix in ("_meanf0", "_histemd", "_wav2vec"):
            if f"tokyo{suffix}" in df_baseline.columns:
                results.append(compute_tarui_preservation(df_baseline, suffix))

    df_comp = pd.DataFrame(results)
    df_comp.to_csv(out_path, index=False)
    print(f"[eval] Comparison table saved: {out_path}")
    return df_comp


def generate_latex_table(df_comp: pd.DataFrame) -> str:
    """
    Generate LaTeX table string from comparison dataframe.

    Parameters
    ----------
    df_comp : pd.DataFrame
        Output of generate_comparison_table()

    Returns
    -------
    str
        LaTeX table code
    """
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Tarui Rate & Balance Ratio & Assessment \\",
        r"\midrule",
    ]
    for _, row in df_comp.iterrows():
        rate = f"{row['tarui_classification_rate']:.1%}"
        ratio = f"{row['mean_tarui_kansai_ratio']:.2f}" if np.isfinite(
            row['mean_tarui_kansai_ratio']) else "---"
        if row.get("tarui_preserved"):
            assessment = "Preserved"
        elif row.get("tarui_overassigned"):
            assessment = "Over-assign"
        elif row.get("tarui_collapsed"):
            assessment = "Collapse"
        else:
            assessment = "Partial"
        lines.append(f"{row['method']} & {rate} & {ratio} & {assessment} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
