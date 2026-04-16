import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from preprocess import preprocess_video, TARGET_FPS, MAX_SHORT_SIDE, VIDEO_EXTS
from detect import detect_objects, init_device, reset_models
from masks import save_tracks
from video_tracker import sam3_text_tracks_from_video, hf_login


def video_hash(video_path):
    return hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()[:12]


def list_videos(root: Path):
    """just find all videos"""
    vids = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            parts = p.relative_to(root).parts
            label = parts[0] if parts[0] in ("real", "fake") else "unknown"
            vids.append((p, label))
    return vids


def detect_and_segment(video_path, frame_paths, top_n=5, max_frames=None):
    """Video-LLaVA detection -> SAM-3 segmentation"""
    prompts = detect_objects(video_path, top_n=top_n)
    print(f"prompts: {prompts}")

    loaded = []
    for p in frame_paths:
        bgr = cv2.imread(p)
        if bgr is not None:
            loaded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frames_rgb = np.stack(loaded) if loaded else None

    tracks, _ = sam3_text_tracks_from_video(video_path, prompts=prompts, frames_rgb=frames_rgb, max_frames=max_frames if frames_rgb is None else None)


    return tracks, prompts


def process_one_video(video_path, label, out_root, top_n, max_frames, skip_existing):
    """preprocess + detect + segment one video, save results"""
    vid_hash = video_hash(video_path)
    vid_out = out_root / vid_hash

    if skip_existing and (vid_out / "meta.json").exists():
        return True

    try:
        frames_dir = str(vid_out / "frames")
        frame_paths, fps, resolution = preprocess_video(video_path, frames_dir, target_fps=TARGET_FPS, max_short_side=MAX_SHORT_SIDE, max_frames=max_frames)


        if len(frame_paths) < 5:
            tqdm.write(f"too few frames ({len(frame_paths)})")
            return False

        tqdm.write(f"preprocessed: {len(frame_paths)} frames, {fps:.0f}fps, {resolution}")

        tracks, prompts = detect_and_segment(video_path, frame_paths, top_n=top_n, max_frames=max_frames)

        if not tracks:
            tqdm.write("no objects tracked")
            return False

        tqdm.write(f"tracked {len(tracks)} objects")

        save_tracks(tracks, prompts, video_path, label, fps, resolution, frame_paths, str(vid_out))

        tqdm.write(f"saved -> {vid_out}")
        return True

    except Exception as e:
        tqdm.write(f"error: {e}")
        return False
    finally:
        gc.collect()


def gpu_worker(gpu_id, num_gpus, videos, out_dir, top_n, max_frames, skip_existing):
    """worker function for each GPU process"""
    reset_models()
    init_device(gpu_id)
    torch.cuda.set_device(gpu_id)

    import video_tracker as _vt
    _vt.DEVICE = torch.device(f"cuda:{gpu_id}")
    _vt._DTYPE = torch.bfloat16
    _vt._sam3_model = None
    _vt._sam3_processor = None

    shard = videos[gpu_id::num_gpus]
    tag = f"[GPU {gpu_id}]"

    print(f"{tag} Processing {len(shard)} videos")
    hf_login()

    ok, fail = 0, 0
    index = {}

    for video_path, label in tqdm(shard, desc=f"GPU {gpu_id}", position=gpu_id):
        tqdm.write(f"{tag} [{ok+fail+1}/{len(shard)}] {video_path.name} ({label})")

        success = process_one_video(str(video_path), label, out_dir, top_n=top_n, max_frames=max_frames, skip_existing=skip_existing)

        index[video_hash(str(video_path))] = {"video_path": str(video_path), "label": label, "success": success}

        if success:
            ok += 1
        else:
            fail += 1

    partial_path = out_dir / f"index_gpu{gpu_id}.json"
    
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"{tag} DONE: {ok} success, {fail} failed out of {len(shard)} videos")


def main():
    p = argparse.ArgumentParser(description="preprocess + segment dataset videos")

    p.add_argument("--data-dir", type=str, required=True, help="input data path")
    p.add_argument("--out-dir", type=str, required=True, help="output directory")
    p.add_argument("--top-n", type=int, default=5, help="number of objects to keep")
    p.add_argument("--max-frames", type=int, default=None, help="limit frames per video")
    p.add_argument("--max-videos", type=int, default=None, help="limit number of videos")
    p.add_argument("--skip-existing", action="store_true", help="skip already processed")
    p.add_argument("--num-gpus", type=int, default=0, help="number of gpus (0 = auto)")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(data_dir)
    if args.max_videos:
        videos = videos[:args.max_videos]

    n_real = sum(1 for _, l in videos if l == "real")
    n_fake = sum(1 for _, l in videos if l == "fake")

    num_gpus = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
    num_gpus = max(1, min(num_gpus, len(videos)))

    if num_gpus <= 1:
        init_device(0)
        gpu_worker(0, 1, videos, out_dir, args.top_n, args.max_frames, args.skip_existing)
    else:
        processes = []
        for gpu_id in range(num_gpus):
            proc = mp.Process(target=gpu_worker, args=(gpu_id, num_gpus, videos, out_dir,args.top_n, args.max_frames, args.skip_existing))
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()


    merged = {}
    for gpu_id in range(num_gpus):
        partial = out_dir / f"index_gpu{gpu_id}.json"

        if partial.exists():
            with open(partial, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
            partial.unlink()

    index_path = out_dir / "index.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    ok = sum(1 for v in merged.values() if v["success"])
    fail = sum(1 for v in merged.values() if not v["success"])
    print(f"DONE: {ok} success, {fail} failed out of {len(videos)} videos")
    print(f"Index: {index_path}")




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
