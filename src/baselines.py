"""
Baseline Distance Metrics

1. Mean F0 Distance: No temporal information
2. Histogram EMD: Distribution only, ignores order
3. wav2vec 2.0: Self-supervised embeddings
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional
import librosa

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]

# wav2vec availability
_HAVE_WAV2VEC = False
_wav2vec_model = None
_wav2vec_processor = None

try:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from sklearn.metrics.pairwise import cosine_distances
    _HAVE_WAV2VEC = True
except ImportError:
    pass


# =============================================================================
# Baseline 1: Mean F0 Distance
# =============================================================================

def mean_f0_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """
    Baseline 1: Absolute difference of mean normalized F0.
    
    This metric discards all temporal information, testing whether
    pitch register alone can distinguish accent types.
    
    Parameters
    ----------
    z1 : np.ndarray
        First normalized pitch track
    z2 : np.ndarray
        Second normalized pitch track
        
    Returns
    -------
    float
        Absolute difference of means
    """
    v1, v2 = np.isfinite(z1), np.isfinite(z2)
    if v1.sum() < 5 or v2.sum() < 5:
        return np.inf
    return abs(np.nanmean(z1[v1]) - np.nanmean(z2[v2]))


# =============================================================================
# Baseline 2: Pitch Histogram + EMD
# =============================================================================

def pitch_histogram(z: np.ndarray, bins: int = 30, zmin: float = -3, zmax: float = 3):
    """
    Compute normalized pitch histogram.
    
    Parameters
    ----------
    z : np.ndarray
        Normalized pitch track
    bins : int
        Number of histogram bins
    zmin : float
        Minimum z-value
    zmax : float
        Maximum z-value
        
    Returns
    -------
    tuple or None
        (histogram, bin_centers) or None if insufficient data
    """
    v = np.isfinite(z)
    if v.sum() < 5:
        return None
    hist, edges = np.histogram(z[v], bins=bins, range=(zmin, zmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist, centers


def histogram_emd_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """
    Baseline 2: Earth Mover's Distance on pitch histograms.
    
    This metric preserves distributional information but ignores
    sequential order, failing to capture temporal dynamics.
    
    Parameters
    ----------
    z1 : np.ndarray
        First normalized pitch track
    z2 : np.ndarray
        Second normalized pitch track
        
    Returns
    -------
    float
        Earth Mover's Distance between histograms
    """
    h1 = pitch_histogram(z1)
    h2 = pitch_histogram(z2)
    if h1 is None or h2 is None:
        return np.inf
    hist1, c1 = h1
    hist2, c2 = h2
    return wasserstein_distance(c1, c2, hist1, hist2)


# =============================================================================
# Baseline 3: wav2vec 2.0 + Cosine
# =============================================================================

def _load_wav2vec():
    """Lazy loading of wav2vec model."""
    global _wav2vec_model, _wav2vec_processor
    if _wav2vec_model is None and _HAVE_WAV2VEC:
        # XLSR-53 multilingual model (base, not fine-tuned)
        # Note: We use the base XLSR-53 for acoustic embedding extraction,
        # not the ASR fine-tuned variant, as we need general acoustic
        # representations rather than phoneme recognition.
        model_name = "facebook/wav2vec2-large-xlsr-53"
        _wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        _wav2vec_model.eval()
        print(f"[wav2vec] Loaded: {model_name}")
    return _wav2vec_processor, _wav2vec_model


def wav2vec_embedding(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extract time-averaged wav2vec 2.0 embedding.
    
    Parameters
    ----------
    path : str
        Path to audio file
    sr : int
        Sample rate
        
    Returns
    -------
    np.ndarray or None
        Time-averaged embedding from last hidden state
    """
    if not _HAVE_WAV2VEC:
        return None
    
    processor, model = _load_wav2vec()
    if model is None:
        return None
    
    try:
        y, _ = librosa.load(path, sr=sr)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    except Exception as e:
        print(f"[wav2vec] error {path}: {e}")
        return None


def wav2vec_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Cosine distance between wav2vec embeddings.
    
    Parameters
    ----------
    emb1 : np.ndarray
        First embedding
    emb2 : np.ndarray
        Second embedding
        
    Returns
    -------
    float
        Cosine distance
    """
    if emb1 is None or emb2 is None:
        return np.inf
    return float(cosine_distances([emb1], [emb2])[0, 0])


# =============================================================================
# Baseline Evaluator
# =============================================================================

class BaselineEvaluator:
    """
    Evaluate all baseline methods on subject files.
    
    Parameters
    ----------
    extractor : TrackExtractor
        F0 extraction module
    refs : dict
        Reference files by accent type
    k_neighbors : int
        Number of nearest neighbors
    """
    
    def __init__(self, extractor, refs: Dict[str, List[str]], k_neighbors: int = 3):
        self.ext = extractor
        self.refs = refs
        self.k = k_neighbors
        
        # Prepare reference tracks and embeddings
        self.ref_tracks = {a: [] for a in ACCENT_TYPES}
        self.ref_wav2vec = {a: [] for a in ACCENT_TYPES}
        
        for a, files in refs.items():
            for f in files:
                z, vr = self.ext.extract(f)
                if z is not None:
                    self.ref_tracks[a].append((f, z))
                
                if _HAVE_WAV2VEC:
                    emb = wav2vec_embedding(f)
                    if emb is not None:
                        self.ref_wav2vec[a].append((f, emb))
    
    def _knn_mean(self, dists: List[float], k: int) -> float:
        """Compute k-NN average distance."""
        finite = [d for d in dists if np.isfinite(d)]
        if not finite:
            return np.inf
        finite.sort()
        return float(np.mean(finite[:min(k, len(finite))]))
    
    def mean_f0_to_class(self, z: np.ndarray, accent_type: str) -> float:
        """Mean F0 distance to accent class."""
        dists = [mean_f0_distance(z, ref_z) 
                 for _, ref_z in self.ref_tracks.get(accent_type, [])]
        return self._knn_mean(dists, self.k)
    
    def hist_emd_to_class(self, z: np.ndarray, accent_type: str) -> float:
        """Histogram EMD to accent class."""
        dists = [histogram_emd_distance(z, ref_z) 
                 for _, ref_z in self.ref_tracks.get(accent_type, [])]
        return self._knn_mean(dists, self.k)
    
    def wav2vec_to_class(self, emb: np.ndarray, accent_type: str) -> float:
        """wav2vec cosine distance to accent class."""
        if emb is None:
            return np.inf
        dists = [wav2vec_cosine_distance(emb, ref_emb) 
                 for _, ref_emb in self.ref_wav2vec.get(accent_type, [])]
        return self._knn_mean(dists, self.k)
    
    def evaluate_subject(self, subj_path: str) -> Dict:
        """
        Evaluate all baselines for one subject.
        
        Parameters
        ----------
        subj_path : str
            Path to subject audio file
            
        Returns
        -------
        dict
            Distances for all methods and accent types
        """
        import os
        z, vr = self.ext.extract(subj_path)
        emb = wav2vec_embedding(subj_path) if _HAVE_WAV2VEC else None
        
        result = {
            'subject': os.path.basename(subj_path),
            'voiced_ratio': vr
        }
        
        for a in ACCENT_TYPES:
            result[f'{a}_meanf0'] = self.mean_f0_to_class(z, a) if z is not None else np.inf
            result[f'{a}_histemd'] = self.hist_emd_to_class(z, a) if z is not None else np.inf
            result[f'{a}_wav2vec'] = self.wav2vec_to_class(emb, a)
        
        return result


def is_wav2vec_available() -> bool:
    """Check if wav2vec 2.0 is available."""
    return _HAVE_WAV2VEC
