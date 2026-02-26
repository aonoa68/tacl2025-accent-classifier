#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced wav2vec 2.0 Baselines for Pitch Accent Classification

This module implements three wav2vec 2.0 configurations to address
concerns about baseline fairness in pitch accent classification:

- Baseline 3a: Time-averaged embeddings (original)
- Baseline 3b: Frame-level DTW on final layer
- Baseline 3c: Frame-level DTW on middle layer (recommended)

Reference:
- Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training
- Pasad et al. (2021). Layer-wise Analysis of a Self-supervised Speech Model
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import warnings

try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch/transformers not available. wav2vec baselines disabled.")

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# =============================================================================
# DTW Implementation for High-dimensional Features
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=False)
    def _dtw_multidim_numba(X: np.ndarray, Y: np.ndarray, 
                            band_ratio: float = 0.15) -> float:
        """
        DTW for multi-dimensional sequences with Sakoe-Chiba band constraint.
        
        Args:
            X: First sequence, shape (T1, D)
            Y: Second sequence, shape (T2, D)
            band_ratio: Sakoe-Chiba band width as ratio of sequence length
            
        Returns:
            Normalized DTW distance
        """
        n, m = X.shape[0], Y.shape[0]
        band = max(int(band_ratio * max(n, m)), 1)
        
        # Initialize cost matrix with infinity
        INF = 1e10
        D = np.full((n + 1, m + 1), INF)
        D[0, 0] = 0.0
        
        # Fill cost matrix with band constraint
        for i in range(1, n + 1):
            j_start = max(1, i - band)
            j_end = min(m + 1, i + band + 1)
            for j in range(j_start, j_end):
                # Euclidean distance for multi-dimensional vectors
                cost = 0.0
                for d in range(X.shape[1]):
                    diff = X[i-1, d] - Y[j-1, d]
                    cost += diff * diff
                cost = np.sqrt(cost)
                
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        
        # Normalize by path length
        return D[n, m] / (n + m)

else:
    def _dtw_multidim_numba(X: np.ndarray, Y: np.ndarray,
                            band_ratio: float = 0.15) -> float:
        """Fallback DTW without Numba acceleration."""
        from scipy.spatial.distance import cdist
        
        n, m = X.shape[0], Y.shape[0]
        band = max(int(band_ratio * max(n, m)), 1)
        
        # Cost matrix
        cost_matrix = cdist(X, Y, metric='euclidean')
        
        INF = 1e10
        D = np.full((n + 1, m + 1), INF)
        D[0, 0] = 0.0
        
        for i in range(1, n + 1):
            j_start = max(1, i - band)
            j_end = min(m + 1, i + band + 1)
            for j in range(j_start, j_end):
                D[i, j] = cost_matrix[i-1, j-1] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        
        return D[n, m] / (n + m)


def dtw_distance_multidim(X: np.ndarray, Y: np.ndarray,
                          band_ratio: float = 0.15) -> float:
    """
    Compute DTW distance between two multi-dimensional sequences.
    
    Args:
        X: First sequence, shape (T1, D)
        Y: Second sequence, shape (T2, D)
        band_ratio: Sakoe-Chiba band width ratio
        
    Returns:
        Normalized DTW distance
    """
    X = np.ascontiguousarray(X.astype(np.float64))
    Y = np.ascontiguousarray(Y.astype(np.float64))
    return _dtw_multidim_numba(X, Y, band_ratio)


# =============================================================================
# Enhanced wav2vec 2.0 Baseline Class
# =============================================================================

class EnhancedWav2VecBaseline:
    """
    Enhanced wav2vec 2.0 baseline with multiple configurations.
    
    Implements three variants:
    - 3a: Time-averaged embeddings (original baseline)
    - 3b: Frame-level DTW on final layer
    - 3c: Frame-level DTW on middle layer
    
    Usage:
        baseline = EnhancedWav2VecBaseline()
        
        # Original baseline (time-averaged)
        d_3a = baseline.compute_distance(audio1, audio2, method="mean_pool", layer=24)
        
        # Frame-level DTW on final layer
        d_3b = baseline.compute_distance(audio1, audio2, method="dtw", layer=24)
        
        # Frame-level DTW on middle layer (recommended)
        d_3c = baseline.compute_distance(audio1, audio2, method="dtw", layer=9)
    """
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-large-xlsr-53",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the enhanced wav2vec baseline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and transformers required for wav2vec baselines")
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = Wav2Vec2Model.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()
        
        # Model info
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded: {self.num_layers} layers, {self.hidden_size} hidden size")
    
    def extract_embeddings(self, 
                           audio: Union[np.ndarray, str, Path],
                           sample_rate: int = 16000,
                           layer: int = -1,
                           return_all_layers: bool = False
                           ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Extract frame-level embeddings from audio.
        
        Args:
            audio: Audio waveform (1D array) or path to audio file
            sample_rate: Sample rate (default 16kHz for wav2vec)
            layer: Layer index to extract (-1 for final, 0 for CNN output)
            return_all_layers: If True, return embeddings from all layers
            
        Returns:
            Frame-level embeddings, shape (T, D)
            or list of embeddings if return_all_layers=True
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(str(audio))
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            audio = waveform.squeeze().numpy()
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        
        # Process
        inputs = self.processor(
            audio, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_values, 
                output_hidden_states=True
            )
        
        if return_all_layers:
            # Return all hidden states (layer 0 = after CNN, 1-24 = transformer layers)
            all_embeddings = []
            for hidden_state in outputs.hidden_states:
                emb = hidden_state.squeeze(0).cpu().numpy()
                all_embeddings.append(emb)
            return all_embeddings
        else:
            # Return specific layer
            if layer == -1:
                layer = self.num_layers
            hidden_state = outputs.hidden_states[layer]
            return hidden_state.squeeze(0).cpu().numpy()
    
    def compute_distance(self,
                         audio1: Union[np.ndarray, str, Path],
                         audio2: Union[np.ndarray, str, Path],
                         method: str = "dtw",
                         layer: int = 9,
                         band_ratio: float = 0.15,
                         sample_rate: int = 16000) -> float:
        """
        Compute distance between two audio samples.
        
        Args:
            audio1: First audio (waveform or path)
            audio2: Second audio (waveform or path)
            method: Distance method ("mean_pool" for 3a, "dtw" for 3b/3c)
            layer: Layer to use (24 for final, 9 for middle)
            band_ratio: DTW band ratio (only for method="dtw")
            sample_rate: Audio sample rate
            
        Returns:
            Distance value
        """
        # Extract embeddings
        emb1 = self.extract_embeddings(audio1, sample_rate, layer)
        emb2 = self.extract_embeddings(audio2, sample_rate, layer)
        
        if method == "mean_pool":
            # Baseline 3a: Time-averaged embeddings + cosine distance
            vec1 = emb1.mean(axis=0)
            vec2 = emb2.mean(axis=0)
            
            # Cosine distance
            cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return 1.0 - cos_sim
        
        elif method == "dtw":
            # Baseline 3b/3c: Frame-level DTW
            return dtw_distance_multidim(emb1, emb2, band_ratio)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_all_layers(self,
                           audio1: Union[np.ndarray, str, Path],
                           audio2: Union[np.ndarray, str, Path],
                           method: str = "dtw",
                           layers: Optional[List[int]] = None,
                           band_ratio: float = 0.15,
                           sample_rate: int = 16000) -> Dict[int, float]:
        """
        Analyze distance across multiple layers.
        
        Args:
            audio1: First audio
            audio2: Second audio
            method: Distance method
            layers: List of layer indices (default: [1,3,6,9,12,15,18,21,24])
            band_ratio: DTW band ratio
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary mapping layer index to distance
        """
        if layers is None:
            layers = [1, 3, 6, 9, 12, 15, 18, 21, 24]
        
        # Extract all layers at once for efficiency
        all_emb1 = self.extract_embeddings(audio1, sample_rate, return_all_layers=True)
        all_emb2 = self.extract_embeddings(audio2, sample_rate, return_all_layers=True)
        
        results = {}
        for layer in layers:
            if layer > len(all_emb1) - 1:
                continue
                
            emb1 = all_emb1[layer]
            emb2 = all_emb2[layer]
            
            if method == "mean_pool":
                vec1 = emb1.mean(axis=0)
                vec2 = emb2.mean(axis=0)
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                results[layer] = 1.0 - cos_sim
            elif method == "dtw":
                results[layer] = dtw_distance_multidim(emb1, emb2, band_ratio)
        
        return results


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def compute_distance_matrix(baseline: EnhancedWav2VecBaseline,
                           audio_files: List[str],
                           method: str = "dtw",
                           layer: int = 9,
                           band_ratio: float = 0.15,
                           show_progress: bool = True) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of audio files.
    
    Args:
        baseline: EnhancedWav2VecBaseline instance
        audio_files: List of audio file paths
        method: Distance method
        layer: Layer to use
        band_ratio: DTW band ratio
        show_progress: Whether to show progress
        
    Returns:
        Distance matrix, shape (N, N)
    """
    n = len(audio_files)
    D = np.zeros((n, n))
    
    # Pre-extract all embeddings
    print("Extracting embeddings...")
    embeddings = []
    for i, f in enumerate(audio_files):
        if show_progress and i % 10 == 0:
            print(f"  {i}/{n}")
        emb = baseline.extract_embeddings(f, layer=layer)
        embeddings.append(emb)
    
    # Compute pairwise distances
    print("Computing distances...")
    total = n * (n - 1) // 2
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if method == "mean_pool":
                vec1 = embeddings[i].mean(axis=0)
                vec2 = embeddings[j].mean(axis=0)
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                d = 1.0 - cos_sim
            else:
                d = dtw_distance_multidim(embeddings[i], embeddings[j], band_ratio)
            
            D[i, j] = d
            D[j, i] = d
            
            count += 1
            if show_progress and count % 100 == 0:
                print(f"  {count}/{total}")
    
    return D


def evaluate_tarui_preservation(baseline: EnhancedWav2VecBaseline,
                                subject_files: List[str],
                                reference_files: Dict[str, List[str]],
                                method: str = "dtw",
                                layer: int = 9,
                                band_ratio: float = 0.15) -> Dict[str, float]:
    """
    Evaluate Tarui preservation rate and balance ratio.
    
    Args:
        baseline: EnhancedWav2VecBaseline instance
        subject_files: List of subject audio files
        reference_files: Dict mapping accent type to list of reference files
        method: Distance method
        layer: Layer to use
        band_ratio: DTW band ratio
        
    Returns:
        Dictionary with 'tarui_rate', 'balance_ratio', 'assignments'
    """
    results = {
        'tarui_count': 0,
        'total_count': 0,
        'tarui_distances': [],
        'kansai_distances': [],
        'assignments': []
    }
    
    for subj_file in subject_files:
        # Compute distances to each reference type
        distances = {}
        for accent_type, ref_files in reference_files.items():
            type_distances = []
            for ref_file in ref_files:
                d = baseline.compute_distance(
                    subj_file, ref_file, 
                    method=method, layer=layer, band_ratio=band_ratio
                )
                type_distances.append(d)
            distances[accent_type] = np.mean(type_distances)
        
        # Find closest type
        closest = min(distances, key=distances.get)
        results['assignments'].append(closest)
        results['total_count'] += 1
        
        if closest == 'tarui':
            results['tarui_count'] += 1
        
        # Track distances for balance ratio
        if 'tarui' in distances:
            results['tarui_distances'].append(distances['tarui'])
        if 'kansai' in distances:
            results['kansai_distances'].append(distances['kansai'])
    
    # Compute metrics
    tarui_rate = results['tarui_count'] / results['total_count'] if results['total_count'] > 0 else 0
    
    if results['tarui_distances'] and results['kansai_distances']:
        balance_ratio = np.mean(results['tarui_distances']) / np.mean(results['kansai_distances'])
    else:
        balance_ratio = float('nan')
    
    return {
        'tarui_rate': tarui_rate,
        'balance_ratio': balance_ratio,
        'assignments': results['assignments'],
        'tarui_count': results['tarui_count'],
        'total_count': results['total_count']
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced wav2vec 2.0 baselines")
    parser.add_argument("--audio1", type=str, help="First audio file")
    parser.add_argument("--audio2", type=str, help="Second audio file")
    parser.add_argument("--method", type=str, default="dtw", 
                        choices=["mean_pool", "dtw"])
    parser.add_argument("--layer", type=int, default=9,
                        help="Layer to use (9 for middle, 24 for final)")
    parser.add_argument("--analyze_layers", action="store_true",
                        help="Analyze all layers")
    parser.add_argument("--model", type=str, 
                        default="facebook/wav2vec2-large-xlsr-53")
    
    args = parser.parse_args()
    
    if not args.audio1 or not args.audio2:
        print("Usage: python wav2vec_enhanced.py --audio1 file1.wav --audio2 file2.wav")
        print("\nExample configurations:")
        print("  Baseline 3a: --method mean_pool --layer 24")
        print("  Baseline 3b: --method dtw --layer 24")
        print("  Baseline 3c: --method dtw --layer 9 (recommended)")
        exit(1)
    
    baseline = EnhancedWav2VecBaseline(model_name=args.model)
    
    if args.analyze_layers:
        print("\nAnalyzing all layers...")
        results = baseline.analyze_all_layers(
            args.audio1, args.audio2, method=args.method
        )
        print("\nLayer-wise distances:")
        for layer, dist in sorted(results.items()):
            print(f"  Layer {layer:2d}: {dist:.4f}")
    else:
        distance = baseline.compute_distance(
            args.audio1, args.audio2,
            method=args.method, layer=args.layer
        )
        print(f"\nDistance ({args.method}, layer {args.layer}): {distance:.4f}")
