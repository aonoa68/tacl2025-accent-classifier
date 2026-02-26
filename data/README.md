# Data

## Audio Data

The speech recordings used in this study are **not publicly available** due to
privacy and ethical constraints. Participants consented to research use only and
did not consent to public distribution of their recordings.

### Data specifications

| Item | Value |
|---|---|
| Speakers | 68 native Japanese speakers |
| Sentences | ~25 read sentences per speaker |
| Sample rate | 16 kHz, 16-bit PCM |
| Recording | Indoor, close-talk microphone |
| Accent types | Tokyo, Kansai, Tarui, Kagoshima |

### For replication

Researchers interested in accessing the audio data for replication purposes
should contact the authors. We will consider reasonable requests on a
case-by-case basis subject to IRB/ethics approval.

### Directory layout (when data is available)

```
data/
├── subjects/          # Subject recordings (*.WAV)
├── references/        # Reference recordings by accent type
│   ├── tokyo_*.wav
│   ├── kansai_*.wav
│   ├── tarui_*.wav
│   └── kagoshima_*.wav  (or none_*.wav)
└── accent_types.xlsx  # Symbolic L/H/R patterns
```

## Symbolic Accent Patterns

`accent_types.xlsx` contains L/H/R pitch-level patterns for each accent type,
used to generate virtual (synthetic) reference contours. This file is included
in the repository.

### Format

| Column 0 | Column 1 | Column 2 | ... |
|---|---|---|---|
| Accent label | Pattern 1 | Pattern 2 | ... |
| tokyo | LHH | LHL | ... |
| kansai | HLL | HHL | ... |
| tarui | LHL | LHR | ... |
| kagoshima | LLL | LLH | ... |

Symbols: **L** = Low, **H** = High, **R** = Rising (transitional).
