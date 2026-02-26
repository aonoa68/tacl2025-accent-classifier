"""
Unsupervised Clustering Analysis

Hierarchical clustering and silhouette analysis for accent space exploration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import MDS
import warnings

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]


def compute_distance_matrix(
    extractor,
    files: List[str],
    distance_func,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distance matrix for a list of audio files.
    
    Parameters
    ----------
    extractor : TrackExtractor
        F0 extraction module
    files : list
        List of audio file paths
    distance_func : callable
        Distance function taking two z-tracks
    show_progress : bool
        Whether to print progress
        
    Returns
    -------
    D : np.ndarray
        Distance matrix (N x N)
    valid_files : list
        List of files that were successfully processed
    """
    # Extract all tracks
    tracks = []
    valid_files = []
    
    for i, f in enumerate(files):
        if show_progress and i % 10 == 0:
            print(f"[unsup] Extracting {i}/{len(files)}")
        z, vr = extractor.extract(f)
        if z is not None and np.isfinite(z).sum() >= 10:
            tracks.append(z)
            valid_files.append(f)
    
    n = len(tracks)
    if n < 2:
        raise ValueError("Need at least 2 valid tracks for clustering")
    
    D = np.zeros((n, n))
    
    total = n * (n - 1) // 2
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(tracks[i], tracks[j])
            D[i, j] = d
            D[j, i] = d
            
            count += 1
            if show_progress and count % 100 == 0:
                print(f"[unsup] Computing distances {count}/{total}")
    
    return D, valid_files


def hierarchical_clustering(
    D: np.ndarray,
    method: str = 'ward',
    k: int = 4
) -> np.ndarray:
    """
    Perform hierarchical clustering on distance matrix.
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix (N x N)
    method : str
        Linkage method ('ward', 'complete', 'average', 'single')
    k : int
        Number of clusters
        
    Returns
    -------
    labels : np.ndarray
        Cluster assignments (0 to k-1)
    """
    # Convert to condensed form for linkage
    condensed = squareform(D, checks=False)
    
    # Perform hierarchical clustering
    Z = linkage(condensed, method=method)
    
    # Cut tree to get k clusters
    labels = fcluster(Z, k, criterion='maxclust') - 1  # 0-indexed
    
    return labels


def analyze_silhouette(
    D: np.ndarray,
    k_range: List[int] = [2, 3, 4, 5, 6]
) -> Dict[int, float]:
    """
    Compute silhouette scores for different numbers of clusters.
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    k_range : list
        List of k values to evaluate
        
    Returns
    -------
    scores : dict
        Mapping from k to silhouette score
    """
    scores = {}
    
    for k in k_range:
        if k >= len(D):
            continue
        
        labels = hierarchical_clustering(D, method='ward', k=k)
        
        # Check if we have at least 2 clusters with members
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            scores[k] = -1.0
            continue
        
        try:
            score = silhouette_score(D, labels, metric='precomputed')
            scores[k] = float(score)
        except Exception as e:
            warnings.warn(f"Silhouette failed for k={k}: {e}")
            scores[k] = -1.0
    
    return scores


def get_cluster_sizes(labels: np.ndarray) -> List[int]:
    """Get size of each cluster, sorted by size descending."""
    unique, counts = np.unique(labels, return_counts=True)
    return sorted(counts, reverse=True)


def compute_mds(D: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """
    Compute MDS embedding from distance matrix.
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    n_components : int
        Number of dimensions
    seed : int
        Random seed
        
    Returns
    -------
    coords : np.ndarray
        MDS coordinates (N x n_components)
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=seed,
        normalized_stress='auto'
    )
    return mds.fit_transform(D)


def find_medoids(D: np.ndarray, labels: np.ndarray) -> List[int]:
    """
    Find medoid (most central point) for each cluster.
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    labels : np.ndarray
        Cluster assignments
        
    Returns
    -------
    medoids : list
        Index of medoid for each cluster
    """
    medoids = []
    
    for label in sorted(np.unique(labels)):
        mask = labels == label
        indices = np.where(mask)[0]
        
        if len(indices) == 1:
            medoids.append(indices[0])
            continue
        
        # Find point with minimum average distance to other cluster members
        sub_D = D[np.ix_(indices, indices)]
        avg_dists = sub_D.mean(axis=1)
        medoid_idx = indices[np.argmin(avg_dists)]
        medoids.append(medoid_idx)
    
    return medoids


def run_unsupervised_analysis(
    extractor,
    files: List[str],
    distance_func,
    k_range: List[int] = [2, 3, 4, 5, 6],
    default_k: int = 4,
    output_prefix: str = "unsup"
) -> Dict:
    """
    Run complete unsupervised analysis pipeline.
    
    Parameters
    ----------
    extractor : TrackExtractor
        F0 extraction module
    files : list
        List of audio file paths
    distance_func : callable
        Distance function
    k_range : list
        Range of k values for silhouette analysis
    default_k : int
        Default number of clusters
    output_prefix : str
        Prefix for output files
        
    Returns
    -------
    results : dict
        Analysis results including:
        - distance_matrix: np.ndarray
        - valid_files: list
        - silhouette_scores: dict
        - labels: np.ndarray (for default_k)
        - cluster_sizes: list
        - mds_coords: np.ndarray
        - medoids: list
    """
    print(f"[unsup] Starting analysis with {len(files)} files")
    
    # Compute distance matrix
    D, valid_files = compute_distance_matrix(extractor, files, distance_func)
    print(f"[unsup] Distance matrix: {D.shape}")
    
    # Silhouette analysis
    silhouette_scores = analyze_silhouette(D, k_range)
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"[unsup] Silhouette scores: {silhouette_scores}")
    print(f"[unsup] Optimal k: {optimal_k} (score: {silhouette_scores[optimal_k]:.3f})")
    
    # Clustering with default_k
    labels = hierarchical_clustering(D, method='ward', k=default_k)
    cluster_sizes = get_cluster_sizes(labels)
    print(f"[unsup] Cluster sizes (k={default_k}): {cluster_sizes}")
    
    # MDS projection
    mds_coords = compute_mds(D)
    
    # Find medoids
    medoids = find_medoids(D, labels)
    
    results = {
        'distance_matrix': D,
        'valid_files': valid_files,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'labels': labels,
        'cluster_sizes': cluster_sizes,
        'mds_coords': mds_coords,
        'medoids': medoids,
        'k': default_k
    }
    
    return results


def interpret_clusters(
    labels: np.ndarray,
    cluster_sizes: List[int]
) -> str:
    """
    Generate linguistic interpretation of cluster structure.
    
    For Kashiwabara (Tarui region) data, we expect:
    - Unimodal structure (one dominant cluster)
    - Not four discrete accent categories
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster assignments
    cluster_sizes : list
        Sizes of each cluster (sorted descending)
        
    Returns
    -------
    interpretation : str
        Linguistic interpretation
    """
    n_total = len(labels)
    n_clusters = len(cluster_sizes)
    
    # Check for dominant cluster
    if cluster_sizes[0] / n_total > 0.6:
        dominant_pct = cluster_sizes[0] / n_total * 100
        interpretation = (
            f"Unimodal structure detected: {cluster_sizes[0]} subjects ({dominant_pct:.1f}%) "
            f"form a single dominant cluster. This is consistent with the linguistic "
            f"homogeneity of the Tarui accent region. The remaining subjects "
            f"({n_total - cluster_sizes[0]}) likely represent individual variation "
            f"or transitional characteristics rather than discrete accent categories."
        )
    else:
        interpretation = (
            f"Multi-modal structure: {n_clusters} clusters with sizes {cluster_sizes}. "
            f"This may indicate genuine accent diversity or methodological artifacts."
        )
    
    return interpretation


# =============================================================================
# Output utilities
# =============================================================================

def save_results(results: Dict, output_dir: str, prefix: str = "unsup"):
    """Save analysis results to files."""
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save distance matrix
    np.save(f"{output_dir}/{prefix}_D_total.npy", results['distance_matrix'])
    
    # Save cluster assignments
    import pandas as pd
    df = pd.DataFrame({
        'file': results['valid_files'],
        'cluster': results['labels']
    })
    df.to_csv(f"{output_dir}/{prefix}_subjects.csv", index=False)
    
    # Save silhouette scores
    with open(f"{output_dir}/{prefix}_silhouette.json", 'w') as f:
        json.dump({
            'silhouette_scores': {f'k{k}': v for k, v in results['silhouette_scores'].items()},
            'optimal_k': results['optimal_k'],
            'cluster_sizes': results['cluster_sizes'],
            'k': results['k'],
            'interpretation': interpret_clusters(results['labels'], results['cluster_sizes'])
        }, f, indent=2)
    
    print(f"[unsup] Results saved to {output_dir}/")
