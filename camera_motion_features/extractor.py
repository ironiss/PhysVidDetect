import hashlib
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from vggt.models.vggt import VGGT #can be installed with `git clone https://github.com/facebookresearch/vggt.git` `cd vggt && pip install -e . && cd ..`
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

cached_model = None
cached_device = None
cached_dtype = None


def load_model():
    """load VGGT once and reuse it"""
    global cached_model, cached_device, cached_dtype
    if cached_model is not None:
        return cached_model, cached_device, cached_dtype

    if torch.cuda.is_available():
        cached_device = "cuda"
        cached_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
    elif torch.backends.mps.is_available():
        cached_device = "mps"
        cached_dtype = torch.float16
    else:
        cached_device = "cpu"
        cached_dtype = torch.float32
        print("no GPU found, running on CPU (slow)")

    cached_model = VGGT.from_pretrained("facebook/VGGT-1B").to(cached_device)
    print("VGGT model loaded")

    return cached_model, cached_device, cached_dtype


def sample_frames(video_path, n_frames):
    """sample frames evenly from a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total<=0:
            raise ValueError(f"Video has no frames: {video_path}")

        if total<n_frames:
            indices = np.arange(total)
        else:
            indices = np.linspace(0, total-1, n_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")

    return np.array(frames), fps


def preprocess_frames(frames, target_size=518):
    """resize frames and turn them into tensors"""
    images = []
    for frame in frames:
        img = Image.fromarray(frame).convert("RGB")
        w, h = img.size

        new_w = target_size
        new_h = max(14, round(h*(new_w/w)/14)*14)

        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        img_tensor = TF.to_tensor(img)

        if new_h>target_size:
            start_y = (new_h - target_size)//2
            img_tensor = img_tensor[:, start_y:start_y + target_size, :]

        images.append(img_tensor)

    return torch.stack(images)


def get_cache_path(video_path, n_frames, cache_dir=None):
    """build cache path for this video and frame count"""
    abs_path = os.path.abspath(video_path)
    key = f"{abs_path}:{n_frames}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    name = f"{Path(video_path).stem}_{n_frames}f_{h}.npz"

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return Path(cache_dir) / name
    return Path(video_path).parent / f".cache_{name}"


def extract_vggt(video_path, n_frames=32, cache_dir=None):
    """ run VGGT or load cached outputs"""
    cache_path = get_cache_path(video_path, n_frames, cache_dir)

    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "extrinsic": data["extrinsic"],
            "intrinsic": data["intrinsic"],
            "depth_map": data["depth_map"],
            "depth_conf": data["depth_conf"],
            "fps": float(data["fps"]),
        }

    model, device, dtype = load_model()
    frames, fps = sample_frames(video_path, n_frames)
    images = preprocess_frames(frames).to(device)

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

    result = {
        "extrinsic": ext4,
        "intrinsic": intrinsic,
        "depth_map": depth_map,
        "depth_conf": depth_conf,
        "fps": np.float64(fps),
    }

    np.savez_compressed(cache_path, **result)
    return result
