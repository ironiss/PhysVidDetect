"""
video_tracker.py
================
Two-stage pipeline: detect main objects → track them through the video.

Stage 1  RT-DETR object detector  (PekingU/rtdetr_r50vd, or swap DETR)
    Samples a few representative frames and runs the detector on each.
    Detections are filtered by score threshold; boxes are ranked by area
    (large box ≈ prominent / main object).  The top-N most frequently
    detected class names are returned as text prompts for SAM-3.

Stage 2  SAM-3  (facebook/sam3)
    Accepts the text prompts and propagates instance masks through the
    video, producing per-object tracks in the format used by
    physics_features.py.

Memory notes
------------
* Detection stage does NOT load the whole video into RAM.
  It seeks to specific frames with cap.set(CAP_PROP_POS_FRAMES) and
  reads only those, so RAM usage is O(n_sample_frames) not O(total_frames).
* SAM-3 loads the full video once internally via load_video().
* If frame JPEGs are already on disk (e.g. saved by the extraction
  pipeline), pass frame_paths= to get_video_tracks() — the detector will
  read sampled JPEGs from disk instead of re-seeking the original file.

Public API
----------
    detect_from_video_path(video_path, ...)   -> list[str]
    detect_from_frame_paths(frame_paths, ...) -> list[str]
    sam3_text_tracks_from_video(video_path, prompts, ...) -> (tracks, frame_sample)
    get_video_tracks(video_path, frame_paths=None, ...) -> (tracks, prompts_used)
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from collections import Counter
from PIL import Image

import supervision as sv
from huggingface_hub import login as _hf_login
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from transformers.video_utils import load_video

# ─────────────────────────────────────────────────────────────
# HuggingFace login (needed for gated models: facebook/sam3)
#
# Set the env-var before running:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# Or call video_tracker.hf_login("hf_xxx") once at startup.
# ─────────────────────────────────────────────────────────────
def hf_login(token: str | None = None) -> None:
    """Log in to HuggingFace Hub. Falls back to the HF_TOKEN env-var."""
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        _hf_login(token=tok, add_to_git_credential=False)
        print("HuggingFace: logged in.")
    else:
        print("HuggingFace: no token found — set HF_TOKEN env-var if models are gated.")

# Auto-login on import if the env-var is present
hf_login()

# ─────────────────────────────────────────────────────────────
# Shared device / dtype
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


# ═════════════════════════════════════════════════════════════
# STAGE 1 — RT-DETR object detector
# ═════════════════════════════════════════════════════════════

# Swap to "facebook/detr-resnet-50" if preferred.
_DET_MODEL_ID = "PekingU/rtdetr_r50vd"

_det_model: AutoModelForObjectDetection | None = None
_det_processor: AutoImageProcessor | None = None


def _load_detector() -> None:
    global _det_model, _det_processor
    if _det_model is not None:
        return
    print(f"Loading detector ({_DET_MODEL_ID}) …")
    _det_processor = AutoImageProcessor.from_pretrained(_DET_MODEL_ID)
    _det_model = (
        AutoModelForObjectDetection
        .from_pretrained(_DET_MODEL_ID)
        .to(DEVICE)
        .eval()
    )
    print("Detector ready.")


def _detect_on_pil(
    image_pil: Image.Image,
    score_thr: float,
    topk_per_frame: int,
) -> list[str]:
    """
    Run the detector on one PIL image.

    Returns the top-k label strings for that frame, ranked by bounding-box
    area (largest box first).  Each label appears at most once (per-frame
    deduplication).  Only detections above ``score_thr`` are considered.
    """
    inputs = _det_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = _det_model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]], device=DEVICE)  # (h, w)
    results = _det_processor.post_process_object_detection(
        outputs, threshold=score_thr, target_sizes=target_sizes
    )[0]

    boxes  = results["boxes"].detach().cpu().numpy()   # (K, 4)
    labels = results["labels"].detach().cpu().numpy()

    if len(boxes) == 0:
        return []

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)[:topk_per_frame]

    seen: set[str] = set()
    frame_labels: list[str] = []
    for j in order:
        name = _det_model.config.id2label[int(labels[j])].lower().strip()
        if name not in seen:
            frame_labels.append(name)
            seen.add(name)

    return frame_labels


def _pick_labels(
    per_frame: list[list[str]],
    top_n: int,
    min_freq: int,
) -> list[str]:
    counter: Counter[str] = Counter()
    for frame_labels in per_frame:
        for name in frame_labels:          # already deduplicated per frame
            counter[name] += 1

    candidates = [lbl for lbl, cnt in counter.most_common() if cnt >= min_freq]
    if not candidates:
        if counter:
            return [counter.most_common(1)[0][0]]
        return ["object"]

    return candidates[:top_n]


def detect_from_video_path(
    video_path: str,
    top_n: int = 5,
    n_sample_frames: int = 5,
    score_thr: float = 0.3,
    topk_per_frame: int = 10,
    min_freq: int = 2,
) -> list[str]:
    """
    Detect main objects by seeking to specific frames — no full video load.

    Uses ``cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`` to jump directly to
    the desired frame indices.

    Parameters
    ----------
    video_path      : path to the video file
    top_n           : max number of distinct class names to return
    n_sample_frames : how many evenly-spaced frames to sample
    score_thr       : minimum detection confidence to keep a box
    topk_per_frame  : max detections kept per frame (by box area)
    min_freq        : label must appear in this many sampled frames to count
    """
    _load_detector()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return ["object"]

    indices = np.linspace(0, total - 1, n_sample_frames, dtype=int).tolist()
    per_frame: list[list[str]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, bgr = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        per_frame.append(_detect_on_pil(pil, score_thr, topk_per_frame))

    cap.release()
    return _pick_labels(per_frame, top_n, min_freq)


def detect_from_frame_paths(
    frame_paths: list[str],
    top_n: int = 5,
    n_sample_frames: int = 5,
    score_thr: float = 0.3,
    topk_per_frame: int = 10,
    min_freq: int = 2,
) -> list[str]:
    """
    Detect main objects from pre-saved JPEG/PNG frame files on disk.

    Used by the extraction pipeline where frames are already saved to disk
    for physics-feature computation — avoids any video re-decoding.

    Parameters are the same as ``detect_from_video_path``.
    """
    _load_detector()

    if not frame_paths:
        return ["object"]

    indices = np.linspace(0, len(frame_paths) - 1, n_sample_frames, dtype=int).tolist()
    per_frame: list[list[str]] = []

    for idx in indices:
        bgr = cv2.imread(frame_paths[idx])
        if bgr is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        per_frame.append(_detect_on_pil(pil, score_thr, topk_per_frame))

    return _pick_labels(per_frame, top_n, min_freq)


# ═════════════════════════════════════════════════════════════
# STAGE 2 — SAM-3 text-prompted video tracker
# ═════════════════════════════════════════════════════════════

_sam3_model: Sam3VideoModel | None = None
_sam3_processor: Sam3VideoProcessor | None = None


def _load_sam3() -> None:
    global _sam3_model, _sam3_processor
    if _sam3_model is not None:
        return
    print("Loading SAM-3 (video segmentation) …")
    _sam3_model = (
        Sam3VideoModel.from_pretrained("facebook/sam3")
        .to(DEVICE, dtype=_DTYPE)
        .eval()
    )
    _sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("SAM-3 ready.")


def sam3_text_tracks_from_video(
    video_path: str,
    prompts: list[str],
    frames_rgb: np.ndarray | None = None,
    max_frames: int | None = None,
    sample_every_seconds: float = 1.0,
) -> tuple[dict[int, list[dict]], list[np.ndarray]]:
    """
    Segment and track every instance of the given object classes.

    Parameters
    ----------
    video_path           : path to the video file (used for fps lookup)
    prompts              : text labels, e.g. ["person", "car"]
    frames_rgb           : optional pre-loaded (T, H, W, 3) uint8 array;
                           if provided, load_video() is skipped entirely
    max_frames           : frame cap applied when loading from video_path
    sample_every_seconds : stride for preview frames returned alongside tracks

    Returns
    -------
    tracks       : dict[obj_id -> list[frame_dict]]
    frame_sample : BGR preview frames (for debug visualisation)
    """
    _load_sam3()

    if frames_rgb is None:
        frames_rgb, _ = load_video(video_path)
        if max_frames is not None:
            frames_rgb = frames_rgb[:max_frames]
    # else: caller already provides pre-loaded, pre-truncated frames

    session = _sam3_processor.init_video_session(
        video=frames_rgb,
        inference_device=DEVICE,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=_DTYPE,
    )
    _sam3_processor.add_text_prompt(session, prompts)

    video_info = sv.VideoInfo.from_video_path(video_path)
    sample_every_n = max(1, int(video_info.fps * sample_every_seconds))

    tracks: dict[int, list[dict]] = {}
    frame_sample: list[np.ndarray] = []

    for mo in _sam3_model.propagate_in_video_iterator(
        inference_session=session,
        max_frame_num_to_track=len(frames_rgb) - 1,
    ):
        out = _sam3_processor.postprocess_outputs(session, mo)
        frame_idx = int(mo.frame_idx)
        object_ids: list[int] = out["object_ids"].tolist()
        masks: np.ndarray = out["masks"].cpu().numpy().astype(bool)  # (K, H, W)

        for i, obj_id in enumerate(object_ids):
            mask = masks[i]
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = float(mask.sum())
            cx, cy = float(xs.mean()), float(ys.mean())
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perim = float(sum(cv2.arcLength(c, True) for c in contours))
            tracks.setdefault(int(obj_id), []).append({
                "t": frame_idx,
                "mask": mask,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "centroid": (cx, cy),
                "perimeter": perim,
            })

        if frame_idx % sample_every_n == 0:
            frame_sample.append(cv2.cvtColor(frames_rgb[frame_idx], cv2.COLOR_RGB2BGR))

    return tracks, frame_sample


# ═════════════════════════════════════════════════════════════
# Combined pipeline
# ═════════════════════════════════════════════════════════════

def get_video_tracks(
    video_path: str,
    frame_paths: list[str] | None = None,
    top_n_objects: int = 5,
    n_sample_frames: int = 5,
    score_thr: float = 0.3,
    topk_per_frame: int = 10,
    min_freq: int = 2,
    max_frames: int | None = None,
    sample_every_seconds: float = 1.0,
) -> tuple[dict[int, list[dict]], list[str]]:
    """
    Full two-stage pipeline: video → tracks + prompts used.

    Stage 1 (RT-DETR) never loads the full video into RAM:
    - If ``frame_paths`` is provided (JPEGs already on disk), reads
      only a few sampled images from disk.
    - Otherwise, seeks to specific frames with cap.set() — O(1) memory.
    Stage 2 (SAM-3) loads the full video exactly once via load_video().

    Parameters
    ----------
    video_path        : path to the video file
    frame_paths       : optional list of pre-decoded JPEG paths on disk;
                        if given, the detector reads from these instead of
                        re-seeking the video (faster + no extra I/O)
    top_n_objects     : max object categories to detect and track
    n_sample_frames   : frames sampled for the detector
    score_thr         : minimum detection confidence to keep a box
    topk_per_frame    : max detections kept per frame (by box area)
    min_freq          : label must appear in ≥ this many sampled frames
    max_frames        : optional SAM-3 frame cap
    sample_every_seconds : preview-frame stride for SAM-3

    Returns
    -------
    tracks       : dict[obj_id -> list[frame_dict]] ready for physics_features.py
    prompts_used : text labels passed to SAM-3
    """
    # ── Stage 1: RT-DETR → text prompts (no full video load) ──
    if frame_paths:
        prompts = detect_from_frame_paths(
            frame_paths,
            top_n=top_n_objects,
            n_sample_frames=n_sample_frames,
            score_thr=score_thr,
            topk_per_frame=topk_per_frame,
            min_freq=min_freq,
        )
    else:
        prompts = detect_from_video_path(
            video_path,
            top_n=top_n_objects,
            n_sample_frames=n_sample_frames,
            score_thr=score_thr,
            topk_per_frame=topk_per_frame,
            min_freq=min_freq,
        )

    print(f"  Detector prompts: {prompts}")

    # ── Stage 2: SAM-3 → mask tracks ─────────────────────────
    # If frame JPEGs are already on disk, load them directly as a
    # numpy array and pass to SAM-3 — skips load_video() which would
    # decode the entire video before truncating to max_frames.
    preloaded: np.ndarray | None = None
    if frame_paths:
        loaded = []
        for p in frame_paths:
            bgr = cv2.imread(p)
            if bgr is not None:
                loaded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if loaded:
            preloaded = np.stack(loaded)   # (T, H, W, 3) uint8

    tracks, _ = sam3_text_tracks_from_video(
        video_path,
        prompts=prompts,
        frames_rgb=preloaded,              # None → falls back to load_video
        max_frames=max_frames if preloaded is None else None,
        sample_every_seconds=sample_every_seconds,
    )

    return tracks, prompts
