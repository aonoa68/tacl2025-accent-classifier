"""
F0 Extraction and Speaker Normalization

Pipeline: Audio → VAD → F0 (YIN/pYIN) → Semitone → MAD-z → Smoothing
"""

import numpy as np
import librosa
from scipy.signal import savgol_filter
from typing import Tuple, Optional


class TrackExtractor:
    """
    F0 extraction with speaker normalization.
    
    Steps:
    1. Voice Activity Detection (VAD)
    2. F0 extraction using YIN (with pYIN fallback)
    3. Short gap interpolation
    4. Semitone conversion
    5. MAD-based robust z-normalization
    6. Savitzky-Golay smoothing
    
    Parameters
    ----------
    sr : int
        Target sample rate (default: 16000)
    fmin : float
        Minimum F0 frequency in Hz (default: 50)
    fmax : float
        Maximum F0 frequency in Hz (default: 700)
    frame : int
        Frame length for F0 extraction (default: 1024)
    hop : int
        Hop length for F0 extraction (default: 160)
    top_db : int
        Threshold for VAD in dB (default: 25)
    min_seg_ms : int
        Minimum segment length in ms (default: 120)
    """
    
    def __init__(
        self,
        sr: int = 16000,
        fmin: float = 50,
        fmax: float = 700,
        frame: int = 1024,
        hop: int = 160,
        top_db: int = 25,
        min_seg_ms: int = 120
    ):
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.frame = frame
        self.hop = hop
        self.top_db = top_db
        self.min_seg_ms = min_seg_ms
        self.cache = {}
    
    def _vad(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Voice Activity Detection using energy-based splitting."""
        intervals = librosa.effects.split(y, top_db=self.top_db)
        segs = []
        min_len = int(sr * (self.min_seg_ms / 1000))
        
        for s, e in intervals:
            if e - s >= min_len:
                segs.append(y[s:e])
        
        return np.concatenate(segs) if segs else y
    
    def _interp_short_gaps(self, f0: np.ndarray, sr: int) -> np.ndarray:
        """Linearly interpolate gaps shorter than 250ms."""
        f0 = f0.copy()
        max_gap = int(0.25 * sr / self.hop)
        n = len(f0)
        i = 0
        
        while i < n:
            if np.isnan(f0[i]):
                j = i
                while j < n and np.isnan(f0[j]):
                    j += 1
                if i > 0 and j < n and (j - i) <= max_gap:
                    f0[i:j] = np.interp(
                        np.arange(i, j),
                        [i - 1, j],
                        [f0[i - 1], f0[j]]
                    )
                i = j
            else:
                i += 1
        
        return f0
    
    def extract(self, path: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract normalized F0 track from audio file.
        
        Parameters
        ----------
        path : str
            Path to audio file
            
        Returns
        -------
        z : np.ndarray or None
            MAD-normalized semitone track (None if extraction failed)
        voiced_ratio : float
            Proportion of voiced frames (0.0-1.0)
        """
        if path in self.cache:
            return self.cache[path]
        
        try:
            y, sr = librosa.load(path, sr=self.sr)
        except Exception as e:
            print(f"[extract] load error {path}: {e}")
            return None, 0.0
        
        # VAD
        y = self._vad(y, sr)
        
        # YIN F0 extraction
        f0 = librosa.yin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            frame_length=self.frame,
            hop_length=self.hop
        )
        f0[f0 <= 0] = np.nan
        vr = float(np.isfinite(f0).mean())
        
        # Fallback to pYIN if voiced ratio is too low
        if vr < 0.10 or np.isfinite(f0).sum() < 10:
            try:
                f0_pyin, vflag, vprob = librosa.pyin(
                    y,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=sr,
                    frame_length=self.frame,
                    hop_length=self.hop
                )
                f0 = f0_pyin.copy()
                use = (vprob > 0.4) & np.isfinite(f0)
                f0[~use] = np.nan
                vr = float(np.isfinite(f0).mean())
            except Exception:
                pass
        
        if vr < 0.03:
            self.cache[path] = (None, vr)
            return self.cache[path]
        
        # Interpolate short gaps
        f0 = self._interp_short_gaps(f0, sr)
        mask = np.isfinite(f0)
        
        if mask.sum() < 10:
            self.cache[path] = (None, vr)
            return self.cache[path]
        
        # Semitone conversion
        f0v = f0[mask]
        fref = np.percentile(f0v, 20)
        st = np.full_like(f0, np.nan, dtype=float)
        st[mask] = 12 * np.log2(np.maximum(f0v, 1e-6) / max(fref, 1e-6))
        
        # MAD-based robust z-normalization
        med = np.nanmedian(st)
        mad = np.nanmedian(np.abs(st - med))
        z = (st - med) / (1.4826 * max(mad, 1e-8))
        
        # Savitzky-Golay smoothing
        if mask.sum() >= 11:
            ztmp = z.copy()
            idx = np.arange(len(z))
            ztmp[~mask] = np.interp(idx[~mask], idx[mask], z[mask])
            zsm = savgol_filter(ztmp, 11, 3, mode="interp")
            z = zsm
            z[~mask] = np.nan
        
        self.cache[path] = (z, vr)
        return self.cache[path]
    
    def clear_cache(self):
        """Clear the extraction cache."""
        self.cache = {}
