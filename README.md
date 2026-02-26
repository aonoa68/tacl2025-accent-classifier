# Speaker-normalized Acoustic Distance for Japanese Pitch Accent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Code and materials for:

> **Speaker-normalized Acoustic Distance for Japanese Pitch Accent:
> Design Principles and Evaluation**
>
> Transactions of the Association for Computational Linguistics (TACL), 2025.

## Overview

Cross-speaker comparison of Japanese pitch accents is unstable due to
individual variation in F0 range, speech rate, and phonation. This work
proposes a **speaker-normalized F0-based distance** using Dynamic Time Warping
(DTW), demonstrating that interpretable distance design can match or exceed
black-box representations for prosodic typology tasks.

### Key contributions

1. **Normalisation pipeline** — Semitone + MAD-based z-scoring that preserves
   phonological contrasts across speakers
2. **Evaluation framework** — Prioritises rank stability and
   intermediate-category preservation (Tarui accent) over raw accuracy
3. **Symbolic injection** — L/H/R accent patterns converted to virtual
   reference contours in acoustic space

### Main results

| Method | Tarui Rate | Balance Ratio | Assessment |
|---|---|---|---|
| Mean F0 | 1.5 % | 1.42 | Collapse |
| Histogram EMD | 85.3 % | 0.83 | Over-assign |
| wav2vec 2.0 | 19.1 % | 0.94 | Partial |
| **DTW (ours)** | **57.4 %** | **1.01** | **Balanced** |

## Installation

```bash
pip install -r requirements.txt
```

Optional (for acceleration and wav2vec baseline):

```bash
pip install numba fastdtw joblib          # DTW acceleration
pip install torch transformers torchaudio  # wav2vec 2.0 baseline
```

## Quick start

### Full pipeline

```bash
python run_pipeline.py \
    --subject_dir   data/subjects \
    --reference_dir data/references \
    --output_dir    outputs/ \
    --excel_path    data/accent_types.xlsx
```

### As a library

```python
from src import TrackExtractor, AccentClassifier, series_distance

ext = TrackExtractor()
z, vr = ext.extract("audio.wav")

clf = AccentClassifier(ext, reference_files)
z, vr, distances = clf.distance_vector("audio.wav")
```

### Benchmarks (Appendix D)

```bash
python benchmark_timing.py
```

### Figures (incl. Appendix C)

```bash
python -m src.visualization
```

## Repository structure

```
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_pipeline.py              # Main entry point
├── benchmark_timing.py          # Computational cost measurement
├── src/
│   ├── __init__.py
│   ├── extractor.py             # F0 extraction & normalisation
│   ├── dtw_distance.py          # DTW distance (static + delta)
│   ├── baselines.py             # Baselines (Mean F0, EMD, wav2vec)
│   ├── classifier.py            # Two-stage accent classifier
│   ├── evaluation.py            # Tarui preservation metrics
│   ├── virtual_reference.py     # Symbolic → acoustic injection
│   ├── unsupervised.py          # Hierarchical clustering analysis
│   ├── wav2vec_enhanced.py      # Enhanced wav2vec baselines (3a/3b/3c)
│   └── visualization.py         # Figure generation
├── configs/
│   └── default.yaml             # Default parameters
├── data/
│   ├── accent_types.xlsx        # Symbolic accent patterns (L/H/R)
│   └── README.md                # Data access instructions
└── outputs/                     # Generated results
```

## Method

### Normalisation pipeline

1. **VAD** — Energy-based voice activity detection
2. **F0** — YIN extraction (pYIN fallback for low-voiced segments)
3. **Semitone** — `s(t) = 12 · log₂(F0(t) / P20(F0))`
4. **MAD-z** — `z(t) = (s(t) − median) / (1.4826 · MAD)`
5. **Smoothing** — Savitzky–Golay filter (window=11, order=3)

### DTW distance

```
D = α · DTW(z) + (1 − α) · DTW(Δz),   α = 0.7
```

Sakoe–Chiba band (ratio=0.15), asymmetric costs (ins=del=1.1, sub=1.0).

### Two-stage classification

- **Stage 1**: {Tokyo, Kagoshima, Kansai-block}
- **Stage 2**: Kansai-block → {Kansai, Tarui}

Data-driven ambiguity thresholds prevent forced classification of borderline
speakers.

## Audio data

The recordings are not publicly distributed (see `data/README.md`).
Researchers wishing to replicate should contact the authors.

## Citation

```bibtex
@article{tacl2025_pitchaccent,
  title   = {Speaker-normalized Acoustic Distance for Japanese Pitch Accent:
             Design Principles and Evaluation},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2025},
}
```

## License

MIT — see [LICENSE](LICENSE).
