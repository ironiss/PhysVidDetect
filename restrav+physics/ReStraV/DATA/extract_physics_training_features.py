"""
extract_physics_training_features.py
=====================================
Extracts 26-dimensional physics features for every real/fake training video
and saves them to physics_features.h5.

Pipeline per video
------------------
1. Florence-2 scans a few frames  →  top-N object class names (text prompts)
2. SAM-3 segments + tracks those objects through all frames  →  tracks dict
3. physics_features.extract_all_physics_features()  →  float32 array (26,)

Output HDF5 layout
------------------
    features   (N, 26) float32   – one row per video
    label      (N,)    int32     – 1 = real, 0 = fake
    path       (N,)    str       – absolute video path
    feat_names (26,)   str       – PHYSICS_FEATURE_KEYS

Usage (run from the DATA/ directory)
--------------------------------------
    cd ~/ReStraV/DATA
    python extract_physics_training_features.py
    python extract_physics_training_features.py --top-n 3 --max-frames 120
"""

import os
import sys
import argparse
import tempfile
import gc
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# ── make ReStraV/ importable ──────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import physics_features as pf
from video_tracker import get_video_tracks        # RAM++ + SAM-3 pipeline
from physics_features import PHYSICS_FEATURE_KEYS # 26 fixed keys

# ── paths ─────────────────────────────────────────────────────
REAL_DIR   = Path("TRAINING_DATA/REAL")
FAKE_DIR   = Path("TRAINING_DATA/FAKE")
OUTPUT_H5  = Path("physics_features.h5")

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")

N_FEATURES = len(PHYSICS_FEATURE_KEYS)   # 26


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def list_videos(root: Path):
    vids = []
    for ext in VIDEO_EXTS:
        vids += sorted(root.rglob(f"*{ext}"))
    return vids


def save_frames(video_path: str, out_dir: str, max_frames: int | None) -> tuple[list[str], float]:
    """Decode video to JPEG frames on disk; return (frame_paths, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = os.path.join(out_dir, f"{i:05d}.jpeg")
        cv2.imwrite(p, frame)
        paths.append(p)
        i += 1
        if max_frames is not None and i >= max_frames:
            break
    cap.release()
    return paths, float(fps)


def process_video(
    video_path: str,
    top_n: int,
    max_frames: int | None,
) -> np.ndarray | None:
    """
    Full pipeline for one video.
    Returns float32 array (26,) or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = os.path.join(tmp, "frames")

        try:
            frame_paths, fps = save_frames(video_path, frames_dir, max_frames)
        except Exception as e:
            tqdm.write(f"  [frame decode error] {e}")
            return None

        if len(frame_paths) < 5:
            tqdm.write(f"  [skip] too few frames ({len(frame_paths)})")
            return None

        try:
            # Stage 1 + 2: RAM++ reads sampled JPEGs already on disk
            # (no second video decode), then SAM-3 loads the original video once.
            tracks, prompts_used = get_video_tracks(
                video_path,
                frame_paths=frame_paths,   # ← avoids re-decoding
                top_n_objects=top_n,
                max_frames=max_frames,
            )
            tqdm.write(f"  prompts={prompts_used}  objects={len(tracks)}")
        except Exception as e:
            tqdm.write(f"  [tracker error] {e}")
            return None

        if not tracks:
            tqdm.write("  [skip] no objects tracked")
            return None

        try:
            feat_vec = pf.extract_all_physics_features(tracks, frame_paths, fps=fps)
        except Exception as e:
            tqdm.write(f"  [feature error] {e}")
            return None

    gc.collect()
    return feat_vec


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract physics features for training videos.")
    parser.add_argument("--top-n",     type=int, default=5,
                        help="Max object categories to detect per video (default 5)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process per video (default: all)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max videos per class (real/fake). E.g. --max-videos 500 "
                             "processes at most 500 real + 500 fake (default: all)")
    args = parser.parse_args()

    real_videos = list_videos(REAL_DIR)
    fake_videos = list_videos(FAKE_DIR)
    if args.max_videos is not None:
        real_videos = real_videos[:args.max_videos]
        fake_videos = fake_videos[:args.max_videos]
    all_videos  = [(str(p), 1) for p in real_videos] + [(str(p), 0) for p in fake_videos]

    print(f"Found {len(real_videos)} real + {len(fake_videos)} fake = {len(all_videos)} videos")
    print(f"Output → {OUTPUT_H5.resolve()}")
    print(f"Feature dim: {N_FEATURES}  ({', '.join(PHYSICS_FEATURE_KEYS[:4])}, ...)")

    # Pre-allocate arrays (fill with NaN so failures are visible)
    feats_arr  = np.full((len(all_videos), N_FEATURES), np.nan, dtype=np.float32)
    labels_arr = np.zeros(len(all_videos), dtype=np.int32)
    paths_arr  = [""] * len(all_videos)

    for idx, (vpath, label) in enumerate(tqdm(all_videos, desc="Videos")):
        tqdm.write(f"\n[{idx+1}/{len(all_videos)}] {Path(vpath).name}")
        feat_vec = process_video(vpath, top_n=args.top_n, max_frames=args.max_frames)
        if feat_vec is not None:
            feats_arr[idx] = feat_vec
        labels_arr[idx] = label
        paths_arr[idx]  = vpath

    # ── save ─────────────────────────────────────────────────
    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_H5, "w") as hf:
        dt_str = h5py.special_dtype(vlen=str)
        hf.create_dataset("features",   data=feats_arr,  dtype="f4")
        hf.create_dataset("label",      data=labels_arr, dtype="i4")
        ds_path = hf.create_dataset("path", (len(all_videos),), dtype=dt_str)
        ds_name = hf.create_dataset("feat_names", (N_FEATURES,), dtype=dt_str)
        for i, p in enumerate(paths_arr):
            ds_path[i] = p
        for i, k in enumerate(PHYSICS_FEATURE_KEYS):
            ds_name[i] = k

    # Summary
    valid = np.isfinite(feats_arr).all(axis=1).sum()
    print(f"\nSaved {len(all_videos)} videos to {OUTPUT_H5}")
    print(f"  Fully valid (no NaN): {valid}/{len(all_videos)}")
    print(f"  NaN rate per feature:")
    for i, k in enumerate(PHYSICS_FEATURE_KEYS):
        nan_pct = 100 * np.isnan(feats_arr[:, i]).mean()
        if nan_pct > 0:
            print(f"    {k}: {nan_pct:.1f}% NaN")


if __name__ == "__main__":
    main()
