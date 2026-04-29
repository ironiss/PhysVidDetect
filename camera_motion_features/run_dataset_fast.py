import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from extractor import sample_frames, preprocess_frames, load_model
from features import extract_features
from visualize import batch_comparison
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
N_FRAMES = 32
DATA_DIR = Path("dataset")
RESULTS_DIR = Path("../feature_data")
PROGRESS_FILE = RESULTS_DIR / "progress.json"


def discover_videos():
    """find videos in dataset and assign rough labels from path"""
    videos = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
            continue

        parts = [p.lower() for p in path.relative_to(DATA_DIR).parts]
        if any(k in p for p in parts for k in ("real", "authentic", "original")):
            label = "real"
        elif any(k in p for p in parts for k in ("fake", "generated", "synthetic", "deepfake")):
            label = "generated"
        else:
            label = "unknown"
        videos.append({
            "path": str(path),
            "name": path.name,
            "label": label,
            "rel_dir": str(path.parent.relative_to(DATA_DIR)),
        })
    return videos


def load_progress():
    """load saved progress if exists"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
        
    return {"completed": {}, "failed": {}, "features": {}}


def save_progress(progress):
    """save progress safely (write + replace)"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=RESULTS_DIR, suffix=".json.tmp")
    
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(progress, f)
        os.replace(tmp_path, PROGRESS_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise


def preload_video(video_path, n_frames):
    """load and preprocess frames on cpu"""
    frames, fps = sample_frames(video_path, n_frames)
    images = preprocess_frames(frames)
    return {"images": images, "fps": fps, "_cached": False}


def gpu_inference(preloaded, model, device, dtype):
    """run VGGT on preloaded frames"""
    if preloaded.get("_cached"):
        return preloaded

    images = preloaded["images"].to(device)
    fps = preloaded["fps"]

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images)
        else:
            images = images.to(dtype)
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = predictions["depth"].squeeze(0).squeeze(-1).cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()

    N = extrinsic.shape[0]
    ext4 = np.zeros((N, 4, 4), dtype=extrinsic.dtype)
    ext4[:, :3, :] = extrinsic
    ext4[:, 3, 3] = 1.0

    return {
        "extrinsic": ext4,
        "intrinsic": intrinsic,
        "depth_map": depth_map,
        "depth_conf": depth_conf,
        "fps": np.float64(fps),
    }





def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    videos = discover_videos()
    print(f"found {len(videos)} videos total")

    label_counts = Counter(v["label"] for v in videos)
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt}")

    progress = load_progress()
    completed_set = set(progress["completed"].keys())
    failed_set = set(progress["failed"].keys())
    remaining = [v for v in videos if v["path"] not in completed_set and v["path"] not in failed_set]
    print(f"already done: {len(completed_set & set(v['path'] for v in videos))}, remaining: {len(remaining)}, failed: {len(failed_set & set(v['path'] for v in videos))}")

    if not remaining:
        print("nothing to process")
        return

    model, device, dtype = load_model()

    executor = ThreadPoolExecutor(max_workers=1)
    next_preload = None
    done = 0
    failed = 0

    pbar = tqdm(remaining, desc="extracting")
    for i, video in enumerate(pbar):
        vpath = video["path"]
        vname = video["name"]
        label = video["label"]

        try:
            if next_preload is not None:
                try:
                    preloaded = next_preload.result()
                except Exception:
                    preloaded = preload_video(vpath, N_FRAMES)
            else:
                preloaded = preload_video(vpath, N_FRAMES)

            if i+1<len(remaining):
                next_preload = executor.submit(preload_video, remaining[i+1]["path"], N_FRAMES)
            else:
                next_preload = None

            vggt_out = gpu_inference(preloaded, model, device, dtype)
            feats = extract_features(vggt_out)

            progress["completed"][vpath] = {"name": vname, "label": label, "features": feats}
            done += 1

        except Exception as e:
            tqdm.write(f"FAILED {vname}: {e}")
            progress["failed"][vpath] = {"name": vname, "label": label, "error": str(e)}
            failed += 1

        pbar.set_postfix(ok=done, fail=failed)

        if (i+1) % 10 == 0 or i == len(remaining)-1:
            save_progress(progress)

    executor.shutdown()

    all_feats = []
    labels = []
    names = []
    for vpath, info in progress["completed"].items():
        all_feats.append(info["features"])
        labels.append(info["label"])
        names.append(info["name"])

    if all_feats:
        batch_comparison(
            all_feats, labels, names,
            output_path=str(RESULTS_DIR / "comparison.png"),
            csv_path=str(RESULTS_DIR / "camera_motion_features.csv"),
        )

    n_ok = len(progress["completed"])
    n_fail = len(progress["failed"])
    print(f"done: {n_ok} completed, {n_fail} failed")


if __name__ == "__main__":
    main()
