import argparse
import csv
import json
import sys
import os
import warnings
from pathlib import Path
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segmentation"))
from physics_features import extract_all_physics_features, PHYSICS_FEATURE_KEYS
from masks import load_tracks


N_FEATURES = len(PHYSICS_FEATURE_KEYS)



def process_one(vid_dir):
    """load one video -- tracks to physics feature vector"""
    vid_dir = str(vid_dir)
    meta_path = os.path.join(vid_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    tracks, frame_paths, _ = load_tracks(vid_dir)
    if not tracks or not frame_paths:
        return None

    try:
        vec = extract_all_physics_features(tracks, frame_paths)
    except Exception as e:
        print(f"error {os.path.basename(vid_dir)}: {e}")
        return None

    label_str = meta.get("label", "unknown")
    label = 1 if label_str == "real" else 0

    return {"features": vec, "label": label, "path": meta.get("video_path", os.path.basename(vid_dir))}




def main():
    parser = argparse.ArgumentParser(description="extract physics features from segmented data")
    parser.add_argument("--data-dir", type=str, required=True, help="path to segmented data")
    parser.add_argument("--out", type=str, default="physics_features.h5", help="output file path")
    parser.add_argument("--workers", type=int, default=0, help="number of worker processes")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    vid_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()])
    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 8)

    results = []
    done = 0
    failed = 0

    with Pool(n_workers) as pool:
        pbar = tqdm(pool.imap_unordered(process_one, vid_dirs), total=len(vid_dirs), desc="extracting")
        for row in pbar:
            if row:
                results.append(row)
                done += 1
            else:
                failed += 1
            pbar.set_postfix(ok=done, fail=failed)

    print(f"done: {done} ok, {failed} failed")

    if not results:
        return

    feats_arr = np.stack([r["features"] for r in results]).astype(np.float32)
    labels_arr = np.array([r["label"] for r in results], dtype=np.int32)
    paths_list = [r["path"] for r in results]

    with h5py.File(args.out, "w") as hf:
        dt_str = h5py.special_dtype(vlen=str)
        hf.create_dataset("features", data=feats_arr, dtype="f4")
        hf.create_dataset("label", data=labels_arr, dtype="i4")
        ds_path = hf.create_dataset("path", (len(results),), dtype=dt_str)
        ds_name = hf.create_dataset("feat_names", (N_FEATURES,), dtype=dt_str)
        for i, p in enumerate(paths_list):
            ds_path[i] = p
        for i, k in enumerate(PHYSICS_FEATURE_KEYS):
            ds_name[i] = k

    csv_path = str(Path(args.out).with_suffix(".csv"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"] + list(PHYSICS_FEATURE_KEYS))
        writer.writeheader()
        for r in results:
            row = {"path": r["path"], "label": r["label"]}
            for k, v in zip(PHYSICS_FEATURE_KEYS, r["features"]):
                row[k] = float(v)
            writer.writerow(row)

    valid = np.isfinite(feats_arr).all(axis=1).sum()
    print(f"saved {len(results)} rows to {args.out} + {csv_path}")
    print(f"fully valid (no NaN): {valid}/{len(results)}")



if __name__ == "__main__":
    main()