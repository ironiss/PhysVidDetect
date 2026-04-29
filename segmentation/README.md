# Segmentation (Video-LLaVA + SAM 3)

Name the prominent objects in each video with Video-LLaVA, then track them frame-by-frame with SAM 3. 

**Output**: per-video frames + packed object masks + `meta.json`.

| File | What |
|---|---|
| `segment_dataset.py` | entry: multi-GPU driver over a dataset (`real/` + `fake/`) |
| `preprocess.py` | decode video, normalize FPS (25) + short side (<=720), dump JPEG frames |
| `detect.py` | Video-LLaVA -> short list of object nouns (the SAM 3 prompts) |
| `video_tracker.py` | SAM 3 text-prompted video tracking -> per-frame object masks |
| `masks.py` | save/load packed masks (`np.packbits`) and `meta.json` |
| `experiments_segmentation.ipynb` | exploratory notebook |

## Run

```
python segment_dataset.py --data-dir ../videos/ --out-dir ../segmented_data/ --num-gpus 8
```

Useful flags: `--top-n 5` (max objects per video), `--max-frames N`, `--skip-existing`.

**Input:** `videos/{real,fake}/*.mp4`

**Output:** `segmented_data/<hash>/{frames/, masks/obj_<id>/<t>.npy, meta.json}` + a merged `index.json`

Requires HF login for gated SAM 3: `huggingface-cli login` (or `HF_TOKEN` env-var).
