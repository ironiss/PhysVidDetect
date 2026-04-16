import json
import os
import tempfile
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

from extractor import extract_vggt
from features import extract_features
from visualize import batch_comparison, single_video_dashboard

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
N_FRAMES = 32
SKIP_DASHBOARDS = os.environ.get("SKIP_DASHBOARDS", "0") == "1"
DATASET_ID = "ironiss/PhysVidDetect-v1"
DATA_DIR = Path("dataset")
RESULTS_DIR = Path("results")
CACHE_DIR = Path("vggt_cache")
PROGRESS_FILE = RESULTS_DIR / "progress.json"


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.mp4")):
        print(f"dataset already present at {DATA_DIR}")
        return

    token = os.environ.get("HF_TOKEN")
    try:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
            token=token,
        )
    except Exception as e:
        print(f"download incomplete: {e}")
        if any(DATA_DIR.rglob("*.mp4")):
            print("continuing with partially downloaded dataset")
        else:
            raise



def discover_videos():
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
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}



def save_progress(progress):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=RESULTS_DIR, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(progress, f)
        os.replace(tmp_path, PROGRESS_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    download_dataset()


    videos = discover_videos()

    label_counts = {}
    for v in videos:
        label_counts[v["label"]] = label_counts.get(v["label"], 0) + 1
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt}")

    progress = load_progress()
    completed_set = set(progress["completed"].keys())
    remaining = [v for v in videos if v["path"] not in completed_set]
    print(f"already done: {len(completed_set)}, remaining: {len(remaining)}, previously failed: {len(progress['failed'])}")

    if not remaining:
        print("nothing to process")
        return

    done = 0
    failed = 0

    pbar = tqdm(remaining, desc="extracting")
    for video in pbar:
        vpath = video["path"]
        vname = video["name"]
        label = video["label"]

        try:
            vggt_out = extract_vggt(vpath, n_frames=N_FRAMES, cache_dir=str(CACHE_DIR))
            feats = extract_features(vggt_out)

            if not SKIP_DASHBOARDS:
                report_dir = RESULTS_DIR / "dashboards" / video["rel_dir"]
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / f"{Path(vname).stem}.png"
                try:
                    single_video_dashboard(vggt_out, feats, str(report_path))
                except Exception:
                    pass

            progress["completed"][vpath] = {"name": vname, "label": label, "features": feats}
            done += 1

        except Exception as e:
            tqdm.write(f"FAILED {vname}: {e}")
            progress["failed"][vpath] = {"name": vname, "label": label, "error": str(e)}
            failed += 1

        pbar.set_postfix(ok=done, fail=failed)
        save_progress(progress)

    all_feats = []
    labels = []
    names = []
    for vpath, info in progress["completed"].items():
        all_feats.append(info["features"])
        labels.append(info["label"])
        names.append(info["name"])

    if all_feats:
        batch_comparison(all_feats, labels, names, output_path=str(RESULTS_DIR / "comparison.png"), csv_path=str(RESULTS_DIR / "features.csv"))

    n_ok = len(progress["completed"])
    n_fail = len(progress["failed"])
    print(f"done: {n_ok} completed, {n_fail} failed")

    if progress["failed"]:
        print("failed videos:")
        for vp, info in progress["failed"].items():
            print(f"  {info['name']}: {info['error']}")




if __name__ == "__main__":
    main()
