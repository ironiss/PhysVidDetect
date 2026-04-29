# Camera motion features (18D)

VGGT-1B over 32 sampled frames -> camera trajectory + depth + intrinsics features.

| File | What |
|---|---|
| `extractor.py` | VGGT load, frame sampling, preprocess, forward (cached to `.npz`) |
| `features.py` | features from VGGT outputs (trajectory, rotation, depth, intrinsics, reprojection) |
| `run_dataset_fast.py` | multi-video driver (thread-prefetch + checkpoints) |
| `run_dataset.py` | older single-pass driver |
| `analyze.py`, `visualize.py` | trajectory/depth plots |

## Run

```
python run_dataset_fast.py
```

**Input:** `/workspace/dataset/{real,fake}/*.mp4`

**Output:** `feature_data/camera_motion_features.csv` (video, label, 18 cols)

VGGT install: `git clone https://github.com/facebookresearch/vggt.git && cd vggt && pip install -e .`
