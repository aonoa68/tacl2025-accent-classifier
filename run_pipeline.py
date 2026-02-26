#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Pipeline: Speaker-normalized Pitch Accent Classification
=============================================================

Usage:
    python run_pipeline.py \\
        --subject_dir data/subjects \\
        --reference_dir data/references \\
        --output_dir outputs/ \\
        --excel_path data/accent_types.xlsx
"""

import os
import math
import argparse
import random
import numpy as np
import pandas as pd

from src.extractor import TrackExtractor
from src.classifier import AccentClassifier, two_stage_classify, calibrate_thresholds
from src.baselines import BaselineEvaluator
from src.evaluation import compute_tarui_preservation, generate_comparison_table
from src.virtual_reference import register_virtual_from_excel
from src.visualization import (
    plot_failure_mode,
    generate_synthetic_failure_example,
    plot_baseline_comparison,
)

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]
SEED = 2025


def list_subject_files(subject_dir: str, exclude=None) -> list:
    """Return sorted list of subject WAV files."""
    import glob

    exclude = exclude or {"68_181018_0216.WAV", "32_2_181010_車戸和子.WAV"}
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    files = set(
        glob.glob(os.path.join(subject_dir, "*.WAV"))
        + glob.glob(os.path.join(subject_dir, "*.wav"))
    )
    return sorted(f for f in files if os.path.basename(f) not in exclude)


def list_reference_files(reference_dir: str) -> dict:
    """Return {accent_type: [paths]} for reference recordings."""
    import glob

    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    refs = {}
    for a in ACCENT_TYPES:
        cand = set(glob.glob(os.path.join(reference_dir, f"{a}*.[Ww][Aa][Vv]")))
        if a == "kagoshima":
            cand |= set(glob.glob(os.path.join(reference_dir, "none*.[Ww][Aa][Vv]")))
        refs[a] = sorted(cand)
    return refs


def main(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Pitch Accent Classification Pipeline")
    print("=" * 60)

    # --- Load files ---
    refs = list_reference_files(args.reference_dir)
    subj = list_subject_files(args.subject_dir)
    print(f"References : {sum(len(v) for v in refs.values())} files")
    print(f"Subjects   : {len(subj)} files")

    # --- Init ---
    ext = TrackExtractor()
    clf = AccentClassifier(ext, refs, k_neighbors=3)

    # --- Virtual references (optional) ---
    if args.excel_path and os.path.exists(args.excel_path):
        print("\nRegistering virtual references from Excel...")
        register_virtual_from_excel(clf, ext, subj, args.excel_path)

    # ========================================================
    # Proposed method (DTW)
    # ========================================================
    print("\n[Step 1] Evaluating proposed method (DTW)...")
    rows, abs_tmp, rel_tmp = [], [], []

    for i, path in enumerate(subj):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(subj)}")
        z, vr, dist = clf.distance_vector(path)
        if z is None:
            rows.append({
                "subject": os.path.basename(path), "voiced_ratio": vr,
                **{f"{a}_distance": np.inf for a in ACCENT_TYPES},
            })
            continue
        vs = sorted(dist.items(), key=lambda x: x[1])
        if len(vs) >= 2 and math.isfinite(vs[0][1]) and math.isfinite(vs[1][1]):
            abs_tmp.append(vs[1][1] - vs[0][1])
            rel_tmp.append((vs[1][1] - vs[0][1]) / max(1e-9, vs[0][1]))
        rows.append({
            "subject": os.path.basename(path), "voiced_ratio": vr,
            **{f"{a}_distance": dist[a] for a in ACCENT_TYPES},
        })

    df_sup = pd.DataFrame(rows)
    amb_abs, amb_rel = calibrate_thresholds(abs_tmp, rel_tmp)
    print(f"  Thresholds: amb_abs={amb_abs:.4f}, amb_rel={amb_rel:.4f}")

    # Classify
    labels, a_m, r_m = [], [], []
    for _, r in df_sup.iterrows():
        dist_d = {a: r[f"{a}_distance"] for a in ACCENT_TYPES}
        finite = {k: v for k, v in dist_d.items() if np.isfinite(v)}
        if len(finite) < 2:
            lab = "unknown" if len(finite) == 0 else "ambiguous"
            labels.append(lab); a_m.append(np.nan); r_m.append(np.nan)
        else:
            lab, ma, mr = two_stage_classify(finite, amb_abs, amb_rel)
            labels.append(lab); a_m.append(ma); r_m.append(mr)

    df_sup["closest_accent"] = labels
    df_sup["margin"] = a_m
    df_sup["rel_margin"] = r_m

    sup_path = os.path.join(args.output_dir, "supervised_results_dtw.csv")
    df_sup.to_csv(sup_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {sup_path}")
    print(f"\n  Classification results:\n{df_sup['closest_accent'].value_counts().to_string()}")

    # ========================================================
    # Baselines
    # ========================================================
    print("\n[Step 2] Evaluating baselines...")
    evaluator = BaselineEvaluator(ext, refs)
    bl_rows = []
    for i, sp in enumerate(subj):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(subj)}")
        bl_rows.append(evaluator.evaluate_subject(sp))

    df_baseline = pd.DataFrame(bl_rows)
    bl_path = os.path.join(args.output_dir, "baseline_results.csv")
    df_baseline.to_csv(bl_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {bl_path}")

    # ========================================================
    # Comparison table
    # ========================================================
    print("\n[Step 3] Generating comparison table...")
    comp_path = os.path.join(args.output_dir, "comparison_table.csv")
    df_comp = generate_comparison_table(df_sup, df_baseline, comp_path)
    print(df_comp.to_string(index=False))

    # ========================================================
    # Figures
    # ========================================================
    print("\n[Step 4] Generating figures...")

    # Failure mode (Appendix C)
    subj_z, tarui_z, kansai_z, fail_dists = generate_synthetic_failure_example()
    plot_failure_mode(subj_z, tarui_z, kansai_z, fail_dists,
                      save_path=os.path.join(fig_dir, "fig_failure_mode.png"))

    # Baseline comparison
    comp_data = {}
    for _, row in df_comp.iterrows():
        comp_data[row["method"]] = {
            "tarui_rate": row["tarui_classification_rate"],
            "balance_ratio": row["mean_tarui_kansai_ratio"],
        }
    if comp_data:
        plot_baseline_comparison(
            comp_data,
            save_path=os.path.join(fig_dir, "fig_baseline_comparison.png"),
        )

    print(f"\nAll outputs saved to: {args.output_dir}")
    return df_sup, df_baseline, df_comp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Japanese Pitch Accent Classification Pipeline"
    )
    parser.add_argument("--subject_dir", required=True,
                        help="Directory of subject WAV files")
    parser.add_argument("--reference_dir", required=True,
                        help="Directory of reference WAV files")
    parser.add_argument("--output_dir", default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--excel_path", default=None,
                        help="Path to accent-type Excel file (optional)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    main(parser.parse_args())
