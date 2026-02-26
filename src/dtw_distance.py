"""
DTW-based Sequence Distance

Implements Sakoe-Chiba constrained DTW with static + delta combination.
"""

import numpy as np
from typing import Optional

# Optional Numba acceleration
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

# Default parameters
BAND_RATIO = 0.15
INS_COST = 1.1
DEL_COST = 1.1
SUB_COST = 1.0
ALPHA = 0.7


if _HAVE_NUMBA:
    @njit(nopython=True, nogil=True)
    def _dtw_avg_cost_numba(a, b, band, ins_c, del_c, sub_c):
        """Numba-accelerated DTW computation."""
        n, m = a.shape[0], b.shape[0]
        D = np.empty((n + 1, m + 1), dtype=np.float64)
        
        for i in range(n + 1):
            for j in range(m + 1):
                D[i, j] = np.inf
        D[0, 0] = 0.0
        
        for i in range(1, n + 1):
            j0 = max(1, i - band)
            j1 = min(m, i + band)
            ai = a[i - 1]
            for j in range(j0, j1 + 1):
                bj = b[j - 1]
                cost = abs(ai - bj) * sub_c
                v = D[i - 1, j] + del_c
                t = D[i, j - 1] + ins_c
                if t < v:
                    v = t
                t = D[i - 1, j - 1] + cost
                if t < v:
                    v = t
                D[i, j] = v
        
        return D[n, m] / float(n + m)


def dtw_avg_cost(
    a: np.ndarray,
    b: np.ndarray,
    band_ratio: float = BAND_RATIO,
    ins_cost: float = INS_COST,
    del_cost: float = DEL_COST,
    sub_cost: float = SUB_COST
) -> float:
    """
    Compute Sakoe-Chiba constrained DTW with average path cost.
    
    Parameters
    ----------
    a : np.ndarray
        First sequence (may contain NaN for unvoiced frames)
    b : np.ndarray
        Second sequence (may contain NaN for unvoiced frames)
    band_ratio : float
        Sakoe-Chiba band width as ratio of sequence length
    ins_cost : float
        Insertion cost
    del_cost : float
        Deletion cost
    sub_cost : float
        Substitution cost multiplier
        
    Returns
    -------
    float
        Average path cost (normalized by path length)
    """
    va, vb = np.isfinite(a), np.isfinite(b)
    if va.sum() < 2 or vb.sum() < 2:
        return np.inf
    
    aa = np.ascontiguousarray(a[va], dtype=np.float64)
    bb = np.ascontiguousarray(b[vb], dtype=np.float64)
    n, m = len(aa), len(bb)
    band = int(max(n, m) * band_ratio)
    
    if _HAVE_NUMBA:
        return float(_dtw_avg_cost_numba(aa, bb, band, ins_cost, del_cost, sub_cost))
    
    # Pure Python fallback
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    
    for i in range(1, n + 1):
        j0, j1 = max(1, i - band), min(m, i + band)
        ai = aa[i - 1]
        for j in range(j0, j1 + 1):
            bj = bb[j - 1]
            cost = abs(ai - bj) * sub_cost
            D[i, j] = min(
                D[i - 1, j] + del_cost,
                D[i, j - 1] + ins_cost,
                D[i - 1, j - 1] + cost
            )
    
    return float(D[n, m] / (n + m))


def delta(seq: np.ndarray) -> np.ndarray:
    """
    Compute delta (first derivative) of a sequence.
    
    Parameters
    ----------
    seq : np.ndarray
        Input sequence (may contain NaN)
        
    Returns
    -------
    np.ndarray
        Delta sequence
    """
    v = np.isfinite(seq)
    if v.sum() < 3:
        return np.full_like(seq, np.nan)
    
    d = np.full_like(seq, np.nan)
    idx = np.where(v)[0]
    
    for k, i in enumerate(idx):
        if k == 0:
            d[i] = seq[idx[1]] - seq[i]
        elif k == len(idx) - 1:
            d[i] = seq[i] - seq[idx[k - 1]]
        else:
            d[i] = (seq[idx[k + 1]] - seq[idx[k - 1]]) / 2.0
    
    return d


def series_distance(
    z1: np.ndarray,
    z2: np.ndarray,
    alpha: float = ALPHA,
    band_ratio: float = BAND_RATIO
) -> float:
    """
    Compute combined DTW distance using static and delta features.
    
    D = α * DTW(z) + (1-α) * DTW(Δz)
    
    Parameters
    ----------
    z1 : np.ndarray
        First normalized pitch track
    z2 : np.ndarray
        Second normalized pitch track
    alpha : float
        Weight for static features (default: 0.7)
    band_ratio : float
        Sakoe-Chiba band width ratio
        
    Returns
    -------
    float
        Combined distance
    """
    ds = dtw_avg_cost(z1, z2, band_ratio=band_ratio)
    d1, d2 = delta(z1), delta(z2)
    dd = dtw_avg_cost(d1, d2, band_ratio=band_ratio)
    
    w_s = float(alpha) if np.isfinite(ds) else 0.0
    w_d = float(1.0 - alpha) if np.isfinite(dd) else 0.0
    denom = w_s + w_d
    
    if denom <= 0.0:
        return np.inf
    
    return float(
        (w_s * (ds if np.isfinite(ds) else 0.0) +
         w_d * (dd if np.isfinite(dd) else 0.0)) / denom
    )
