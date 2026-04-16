import json
import os
import cv2
import numpy as np


def save_tracks(tracks, prompts, video_path, label, fps, resolution, frame_paths, out_dir):
    """save masks as .npy files and metadata as meta.json"""
    os.makedirs(out_dir, exist_ok=True)
    masks_root = os.path.join(out_dir, "masks")

    object_ids = []

    for obj_id, seq in tracks.items():
        obj_dir = os.path.join(masks_root, f"obj_{obj_id}")
        os.makedirs(obj_dir, exist_ok=True)
        object_ids.append(int(obj_id))

        for d in seq:
            t = int(d["t"])
            mask_path = os.path.join(obj_dir, f"{t:05d}.npy")
            np.save(mask_path, np.packbits(d["mask"]))

    meta = {
        "video_path": os.path.abspath(video_path),
        "label": label,
        "fps": fps,
        "resolution": list(resolution),
        "prompts": prompts,
        "object_ids": sorted(object_ids),
        "n_frames": len(frame_paths),
        "n_objects": len(tracks),
        "mask_shape": list(tracks[object_ids[0]][0]["mask"].shape) if object_ids and tracks[object_ids[0]] else [],
    }

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def load_tracks(saved_dir):
    """load previously saved masks + metadata"""
    meta_path = os.path.join(saved_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    frames_dir = os.path.join(saved_dir, "frames")
    frame_paths = sorted(os.path.join(frames_dir, fn) for fn in os.listdir(frames_dir) if fn.endswith(".jpeg"))

    mask_shape = tuple(meta["mask_shape"])
    masks_root = os.path.join(saved_dir, "masks")

    tracks = {}
    for obj_id in meta["object_ids"]:
        obj_dir = os.path.join(masks_root, f"obj_{obj_id}")
        if not os.path.isdir(obj_dir):
            continue

        seq = []
        for fn in sorted(os.listdir(obj_dir)):
            if not fn.endswith(".npy"):
                continue
            
            t = int(fn.replace(".npy", ""))
            packed = np.load(os.path.join(obj_dir, fn))
            mask = np.unpackbits(packed)[:mask_shape[0] * mask_shape[1]].reshape(mask_shape).astype(bool)

            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = float(mask.sum())
            cx, cy = float(xs.mean()), float(ys.mean())
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perim = float(sum(cv2.arcLength(c, True) for c in contours))

            seq.append({
                "t": t,
                "mask": mask,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "centroid": (cx, cy),
                "perimeter": perim,
            })

        if seq:
            tracks[obj_id] = sorted(seq, key=lambda d: d["t"])

    return tracks, frame_paths, meta["fps"]
