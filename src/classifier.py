"""
Two-Stage Accent Classifier

Stage 1: {Tokyo, Kagoshima, Kansai-block}
Stage 2: Kansai-block → {Kansai, Tarui}
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .dtw_distance import series_distance, ALPHA, BAND_RATIO

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]

# Classification thresholds
AMB_ABS_FLOOR = 0.01
AMB_REL_FLOOR = 0.03
TARGET_AMBIGUOUS_RATE = 0.25


class AccentClassifier:
    """
    Two-stage k-NN accent classifier using DTW distance.
    
    Parameters
    ----------
    extractor : TrackExtractor
        F0 extraction and normalization module
    refs : dict
        Reference files by accent type {accent_type: [file_paths]}
    k_neighbors : int
        Number of nearest neighbors for distance averaging
    alpha : float
        Weight for static vs delta in DTW distance
    band_ratio : float
        Sakoe-Chiba band width ratio
    """
    
    def __init__(
        self,
        extractor,
        refs: Dict[str, List[str]],
        k_neighbors: int = 3,
        alpha: float = ALPHA,
        band_ratio: float = BAND_RATIO
    ):
        self.ext = extractor
        self.refs = refs
        self.k = k_neighbors
        self.alpha = alpha
        self.band = band_ratio
        
        # Extract reference tracks
        self.ref_tracks = {a: [] for a in ACCENT_TYPES}
        for a, files in refs.items():
            for f in files:
                z, vr = self.ext.extract(f)
                if z is not None:
                    self.ref_tracks[a].append((f, z, vr))
        
        for a, lst in self.ref_tracks.items():
            print(f"[refs] {a}: usable={len(lst)} / total={len(self.refs.get(a, []))}")
    
    def distance_to_class(self, subj_track: np.ndarray, accent_type: str) -> float:
        """
        Compute k-NN average distance from subject to accent class.
        
        Parameters
        ----------
        subj_track : np.ndarray
            Subject's normalized pitch track
        accent_type : str
            Target accent type
            
        Returns
        -------
        float
            Average distance to k nearest references
        """
        cands = self.ref_tracks.get(accent_type, [])
        if not cands:
            return np.inf
        
        dists = []
        for _, tr, _ in cands:
            d = series_distance(subj_track, tr, alpha=self.alpha, band_ratio=self.band)
            if np.isfinite(d):
                dists.append(d)
        
        if not dists:
            return np.inf
        
        dists.sort()
        return float(np.mean(dists[:min(self.k, len(dists))]))
    
    def distance_vector(self, subj_path: str) -> Tuple[Optional[np.ndarray], float, Dict[str, float]]:
        """
        Compute distance vector from subject to all accent classes.
        
        Parameters
        ----------
        subj_path : str
            Path to subject audio file
            
        Returns
        -------
        z : np.ndarray or None
            Subject's normalized pitch track
        vr : float
            Voiced ratio
        distances : dict
            Distance to each accent type
        """
        z, vr = self.ext.extract(subj_path)
        if z is None:
            return None, vr, {a: np.inf for a in ACCENT_TYPES}
        return z, vr, {a: self.distance_to_class(z, a) for a in ACCENT_TYPES}
    
    def add_virtual_reference(self, accent_type: str, z_track: np.ndarray):
        """
        Add a virtual (synthetic) reference track.
        
        Parameters
        ----------
        accent_type : str
            Accent type for the virtual reference
        z_track : np.ndarray
            Synthesized normalized pitch track
        """
        if accent_type not in self.ref_tracks:
            self.ref_tracks[accent_type] = []
        vr = float(np.isfinite(z_track).mean())
        self.ref_tracks[accent_type].append(
            (f"VIRT_{accent_type}_{len(self.ref_tracks[accent_type])}", z_track, vr)
        )


def _top2_finite(dmap: Dict[str, float]) -> Optional[Tuple]:
    """Get top 2 finite distances."""
    items = [(k, v) for k, v in dmap.items() if np.isfinite(v)]
    if len(items) < 2:
        return None
    items.sort(key=lambda x: x[1])
    return items[0], items[1]


def two_stage_classify(
    dist_dict: Dict[str, float],
    amb_abs: float = AMB_ABS_FLOOR,
    amb_rel: float = AMB_REL_FLOOR
) -> Tuple[str, float, float]:
    """
    Two-stage classification with ambiguity detection.
    
    Stage 1: {Tokyo, Kagoshima, Kansai-block}
    Stage 2: Kansai-block → {Kansai, Tarui}
    
    Parameters
    ----------
    dist_dict : dict
        Distance to each accent type
    amb_abs : float
        Absolute margin threshold for ambiguity
    amb_rel : float
        Relative margin threshold for ambiguity
        
    Returns
    -------
    label : str
        Predicted accent type or 'ambiguous'/'unknown'
    abs_margin : float
        Absolute margin between top 2 distances
    rel_margin : float
        Relative margin between top 2 distances
    """
    # Stage 1: coarse classification
    cand1 = {}
    if 'tokyo' in dist_dict:
        cand1['tokyo'] = dist_dict['tokyo']
    if 'kagoshima' in dist_dict:
        cand1['kagoshima'] = dist_dict['kagoshima']
    
    ks = [dist_dict[x] for x in ('kansai', 'tarui')
          if x in dist_dict and np.isfinite(dist_dict[x])]
    if ks:
        cand1['kansai_block'] = min(ks)
    
    top2 = _top2_finite(cand1)
    if top2 is None:
        return 'unknown', np.nan, np.nan
    
    (w1, d1), (_, d2) = top2
    abs_m1 = d2 - d1
    rel_m1 = abs_m1 / max(1e-9, d1)
    
    if abs_m1 < amb_abs or rel_m1 < amb_rel:
        return 'ambiguous', abs_m1, rel_m1
    
    if w1 != 'kansai_block':
        return w1, abs_m1, rel_m1
    
    # Stage 2: Kansai-block refinement
    cand2 = {k: dist_dict[k] for k in ('kansai', 'tarui')
             if k in dist_dict and np.isfinite(dist_dict[k])}
    
    if len(cand2) == 0:
        return 'unknown', np.nan, np.nan
    if len(cand2) == 1:
        return list(cand2.keys())[0], np.inf, np.inf
    
    (w2, d1b), (_, d2b) = sorted(cand2.items(), key=lambda x: x[1])[:2]
    abs_m2 = d2b - d1b
    rel_m2 = abs_m2 / max(1e-9, d1b)
    
    # Stricter thresholds for Stage 2
    if (abs_m2 >= max(0.015, 0.75 * amb_abs)) and (rel_m2 >= max(0.04, 0.75 * amb_rel)):
        return w2, abs_m2, rel_m2
    
    # Tie-break: assign to broader category (Kansai)
    return 'kansai', abs_m2, rel_m2


def calibrate_thresholds(
    abs_margins: List[float],
    rel_margins: List[float],
    floor_abs: float = AMB_ABS_FLOOR,
    floor_rel: float = AMB_REL_FLOOR,
    target_rate: float = TARGET_AMBIGUOUS_RATE
) -> Tuple[float, float]:
    """
    Automatically calibrate ambiguity thresholds from data.
    
    Parameters
    ----------
    abs_margins : list
        Absolute margins from preliminary classification
    rel_margins : list
        Relative margins from preliminary classification
    floor_abs : float
        Minimum absolute threshold
    floor_rel : float
        Minimum relative threshold
    target_rate : float
        Target ambiguous rate
        
    Returns
    -------
    th_abs : float
        Calibrated absolute threshold
    th_rel : float
        Calibrated relative threshold
    """
    va = np.array([x for x in abs_margins if np.isfinite(x)])
    vr = np.array([x for x in rel_margins if np.isfinite(x)])
    
    th_abs = max(
        np.quantile(va, min(0.5, target_rate)) if len(va) >= 5 else floor_abs,
        floor_abs
    )
    th_rel = max(
        np.quantile(vr, min(0.5, target_rate)) if len(vr) >= 5 else floor_rel,
        floor_rel
    )
    
    return float(th_abs), float(th_rel)
