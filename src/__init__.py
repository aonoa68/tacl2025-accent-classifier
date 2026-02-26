"""
Speaker-normalized Acoustic Distance for Japanese Pitch Accent
TACL 2025
"""

__version__ = "2.0.0"

from .extractor import TrackExtractor
from .dtw_distance import series_distance, dtw_avg_cost, delta
from .classifier import AccentClassifier, two_stage_classify, calibrate_thresholds
from .baselines import BaselineEvaluator, mean_f0_distance, histogram_emd_distance
from .evaluation import compute_tarui_preservation, generate_comparison_table
from .virtual_reference import register_virtual_from_excel

__all__ = [
    "TrackExtractor",
    "series_distance",
    "dtw_avg_cost",
    "delta",
    "AccentClassifier",
    "two_stage_classify",
    "calibrate_thresholds",
    "BaselineEvaluator",
    "mean_f0_distance",
    "histogram_emd_distance",
    "compute_tarui_preservation",
    "generate_comparison_table",
    "register_virtual_from_excel",
]
