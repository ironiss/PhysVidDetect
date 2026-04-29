# Physics features (26D)

Object-based features from SAM 3 masks + raw frames. CPU only, no GPU.

| File | What |
|---|---|
| `extract_features.py` | entry: multiprocess driver, H5/CSV output |
| `physics_features.py` | core: features (`PHYSICS_FEATURE_KEYS`) -- inter-object, boundary, edge-flow, inside-out, HF noise, PSD, valid frac |
| `helpers.py` | shared utils (safe_div, sobel_mag_theta, make_ring, etc.) |

## Run

```
python extract_features.py --data-dir ../segmented_data/ --workers 8
```

**Input:** `segmented_data/<hash>/` produced by `segmentation/`

**Output:** `feature_data/object_based_features.h5` + `.csv` (path, label, 26 cols)
