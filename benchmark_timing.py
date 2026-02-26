#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational Cost Benchmark (Appendix D)
==========================================

Measures per-pair computation time for each distance metric.

Usage:
    python benchmark_timing.py
"""

import time
import numpy as np
from src.dtw_distance import dtw_avg_cost, series_distance
from src.baselines import mean_f0_distance, histogram_emd_distance


def generate_synthetic_f0(length: int = 500, seed: int = 0) -> np.ndarray:
    """Generate a synthetic F0 contour for benchmarking."""
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    z = -0.5 * t + 0.8 * np.sin(2 * np.pi * t * 3) + np.random.normal(0, 0.2, length)
    z[np.random.random(length) >= 0.85] = np.nan
    return z


def run_benchmark(n_pairs: int = 30, track_length: int = 500,
                  n_timing_runs: int = 50) -> dict:
    """
    Benchmark per-pair computation time (ms) for each distance.

    Parameters
    ----------
    n_pairs : int
        Number of random pairs to time.
    track_length : int
        Length of each synthetic F0 contour (frames).
    n_timing_runs : int
        Number of repeated runs per pair for stable timing.

    Returns
    -------
    dict
        {method_name: mean_time_ms}
    """
    pairs = [
        (generate_synthetic_f0(track_length, i),
         generate_synthetic_f0(track_length, i + 1000))
        for i in range(n_pairs)
    ]

    methods = [
        ("Mean F0", mean_f0_distance),
        ("Histogram EMD", histogram_emd_distance),
        ("DTW (static)", dtw_avg_cost),
        ("DTW (full: z + Δz)", series_distance),
    ]

    results = {}
    for label, fn in methods:
        times = []
        for z1, z2 in pairs:
            t0 = time.perf_counter()
            for _ in range(n_timing_runs):
                fn(z1, z2)
            elapsed = (time.perf_counter() - t0) / n_timing_runs
            times.append(elapsed)
        ms = float(np.mean(times) * 1000)
        results[label] = ms

    return results


def scaling_analysis(lengths=(100, 200, 500, 1000, 2000),
                     n_pairs: int = 10, n_runs: int = 20) -> dict:
    """
    Measure how DTW cost scales with sequence length.

    Returns
    -------
    dict
        {length: time_ms}
    """
    results = {}
    for L in lengths:
        pairs = [
            (generate_synthetic_f0(L, i), generate_synthetic_f0(L, i + 500))
            for i in range(n_pairs)
        ]
        times = []
        for z1, z2 in pairs:
            t0 = time.perf_counter()
            for _ in range(n_runs):
                series_distance(z1, z2)
            times.append((time.perf_counter() - t0) / n_runs)
        results[L] = float(np.mean(times) * 1000)
    return results


if __name__ == "__main__":
    from src.dtw_distance import _HAVE_NUMBA

    print("=" * 60)
    print("COMPUTATIONAL COST BENCHMARK")
    print(f"Numba available: {_HAVE_NUMBA}")
    print("=" * 60)

    bench = run_benchmark()
    print(f"\n{'Method':<25s} {'Time (ms)':>10s}")
    print("-" * 36)
    for method, t in bench.items():
        print(f"{method:<25s} {t:>10.4f}")

    print("\n" + "=" * 60)
    print("SCALING ANALYSIS (DTW full)")
    print("=" * 60)
    scaling = scaling_analysis()
    print(f"\n{'Length':<10s} {'Time (ms)':>10s}")
    print("-" * 21)
    for L, t in scaling.items():
        print(f"{L:<10d} {t:>10.4f}")
