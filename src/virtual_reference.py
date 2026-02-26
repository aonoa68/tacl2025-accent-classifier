"""
Symbolic Virtual Reference Generation

Convert L/H/R patterns from Excel to synthetic F0 contours.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, List, Optional

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]


def _norm_label(s: str) -> str:
    """Normalize accent type label."""
    if s is None:
        return ""
    t = str(s).strip().lower()
    if "tokyo" in t or "東京" in t:
        return "tokyo"
    if "kansai" in t or "京阪" in t or "関西" in t:
        return "kansai"
    if "tarui" in t or "垂井" in t:
        return "tarui"
    if "kagoshima" in t or "鹿児島" in t or "none" in t:
        return "kagoshima"
    return ""


def pattern_to_levels(patt: str, low: float = -1.0, high: float = +1.0) -> Optional[List[float]]:
    """
    Convert L/H/R pattern string to pitch level sequence.
    
    Symbol mapping:
    - L (Low): -1.0 (phrase-initial or post-accent low)
    - H (High): +1.0 (accent peak or phrase-level high)
    - R (Rising): +0.5 (transitional rise, accent nucleus approach)
    
    Parameters
    ----------
    patt : str
        Pattern string like "LHH" or "HLL"
    low : float
        Value for L symbol
    high : float
        Value for H symbol
        
    Returns
    -------
    list or None
        List of pitch levels, or None if pattern is invalid
    """
    if not isinstance(patt, str) or not patt.strip():
        return None
    
    mapping = {'L': low, 'H': high, 'R': (low + high) / 2.0}
    seq = []
    
    for ch in patt.strip().upper():
        if ch in mapping:
            seq.append(mapping[ch])
        elif ch.isspace():
            continue
        else:
            return None
    
    return seq if seq else None


def synth_from_pattern(
    donor_z: np.ndarray,
    levels: List[float],
    tilt: float = 0.0,
    jitter: float = 0.04,
    smooth: int = 7
) -> np.ndarray:
    """
    Synthesize F0 contour from L/H/R levels.
    
    The synthetic contour inherits the voiced/unvoiced structure
    from a donor recording, ensuring realistic frame patterns.
    
    Parameters
    ----------
    donor_z : np.ndarray
        Donor normalized pitch track (provides voiced mask)
    levels : list
        Pitch level sequence from pattern_to_levels()
    tilt : float
        Linear declination slope (typically -0.05 to -0.10)
    jitter : float
        Standard deviation of random pitch perturbation
    smooth : int
        Savitzky-Golay filter window (odd number)
        
    Returns
    -------
    np.ndarray
        Synthesized normalized pitch track
    """
    T = len(donor_z)
    M = len(levels)
    z = np.full(T, np.nan, dtype=float)
    
    # Equal mora boundaries
    bd = np.linspace(0, T, M + 1).astype(int)
    bd[-1] = T
    
    for m in range(M):
        s, e = bd[m], bd[m + 1]
        if e > s:
            z[s:e] = levels[m]
    
    # Apply declination tilt
    z += tilt * np.linspace(0, 1, T)
    
    # Add jitter for naturalness
    if jitter > 0:
        z += np.random.normal(0, jitter, T)
    
    # Apply donor's voiced mask
    mask = np.isfinite(donor_z)
    z[~mask] = np.nan
    
    # Savitzky-Golay smoothing
    if smooth and np.isfinite(z).sum() >= smooth:
        ztmp = z.copy()
        idx = np.arange(T)
        valid = np.isfinite(ztmp)
        if valid.sum() >= 2:
            ztmp[~valid] = np.interp(
                idx[~valid],
                idx[valid],
                ztmp[valid]
            )
            ztmp = savgol_filter(ztmp, smooth, 3, mode="interp")
            z = ztmp
            z[~mask] = np.nan
    
    return z


def register_virtual_from_excel(
    classifier,
    extractor,
    subject_files: List[str],
    excel_path: str,
    per_row_variants: int = 1,
    tilt_by_label: Optional[Dict[str, float]] = None,
    seed: int = 2025
) -> int:
    """
    Generate and register virtual references from Excel file.
    
    The Excel file should have:
    - Column 0: Accent type label (e.g., "tokyo", "東京")
    - Columns 1+: L/H/R pattern strings (e.g., "LHH", "HLL")
    
    Parameters
    ----------
    classifier : AccentClassifier
        Classifier to add virtual references to
    extractor : TrackExtractor
        F0 extraction module
    subject_files : list
        List of subject audio paths (for donor extraction)
    excel_path : str
        Path to Excel file with patterns
    per_row_variants : int
        Number of variants to generate per pattern
    tilt_by_label : dict, optional
        Declination tilt by accent type
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    int
        Number of virtual references created
    """
    import os
    
    if not os.path.exists(excel_path):
        print(f"[virt-excel] not found: {excel_path}")
        return 0
    
    df = pd.read_excel(excel_path, sheet_name=0)
    if df.shape[1] < 2:
        print("[virt-excel] need at least 2 columns")
        return 0
    
    # Get donor track
    donor_z = None
    for p in subject_files[:30]:
        z, vr = extractor.extract(p)
        if z is not None and np.isfinite(z).sum() >= 20:
            donor_z = z
            break
    
    if donor_z is None:
        print("[virt-excel] donor extraction failed")
        return 0
    
    rng = np.random.RandomState(seed + 2468)
    made = 0
    
    default_tilt = {
        'tokyo': -0.05,
        'kansai': -0.10,
        'tarui': -0.08,
        'kagoshima': 0.00
    }
    tilt_map = tilt_by_label if tilt_by_label else default_tilt
    
    for ridx, row in df.iterrows():
        lab = _norm_label(row.iloc[0])
        if lab not in ACCENT_TYPES:
            continue
        
        patterns = [c for c in row.iloc[1:].values if isinstance(c, str) and c.strip()]
        if not patterns:
            continue
        
        tilt = float(tilt_map.get(lab, 0.0))
        
        for patt in patterns[:5]:  # Limit patterns per row
            lv = pattern_to_levels(patt)
            if lv is None:
                continue
            
            for _ in range(per_row_variants):
                zsyn = synth_from_pattern(
                    donor_z,
                    lv,
                    tilt=tilt,
                    jitter=0.04 * (1 + rng.randn() * 0.1),
                    smooth=7
                )
                classifier.add_virtual_reference(lab, zsyn)
                made += 1
        
        print(f"[virt-excel] +{lab} row={ridx}")
    
    print(f"[virt-excel] Total synthetic refs: {made}")
    return made
