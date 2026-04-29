# Latent features (80D)

SD 1.5 VAE + DDIM inversion -> z0 + eps features.

| File | What |
|---|---|
| `extract_latent_features.py` | entry: SD pipeline, DDIM, dataset loop, H5/CSV output |
| `combined_features.py` | orchestrator: combines z0 + eps |
| `latent_features.py` | z0 features (stats, channels, FFT, temporal, manifold, object-level) |
| `noise_features.py` | eps features (temporal, frequency, spatial) |
| `config.py` | `LATENT_FEATURE_KEYS`, `NOISE_FEATURE_KEYS`, `ALL_FEATURE_KEYS` |

## Run

```
python extract_latent_features.py
```

**Input:** `segmented_data/<hash>/{frames/, masks/, meta.json}`

**Output:** `feature_data/latent_noise_festures.h5` + `.csv` (path, label, 80 cols)
