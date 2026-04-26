
import os
import cv2
import numpy as np
import torch
import supervision as sv
from huggingface_hub import login as hf_login_fn
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video


def hf_login(token= None):
    """Log in to HuggingFace Hub. Falls back to the HF_TOKEN env-var."""
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        hf_login_fn(token=tok, add_to_git_credential=False)
        print("HuggingFace: logged in.")
    else:
        print("HuggingFace: no token found -- set HF_TOKEN env-var if models are gated.")

hf_login()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

sam3_model= None
sam3_processor= None


def load_sam3():
    global sam3_model, sam3_processor
    if sam3_model is not None:
        return


    sam3_model = (Sam3VideoModel.from_pretrained("facebook/sam3").to(DEVICE, dtype=DTYPE).eval())
    sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")



def sam3_text_tracks_from_video(video_path, prompts, frames_rgb: np.ndarray | None = None, max_frames= None, sample_every_seconds= 1.0):
    """run sam3 with text prompts and collect object tracks"""
    load_sam3()

    if frames_rgb is None:
        frames_rgb, _ = load_video(video_path)

        if max_frames is not None:
            frames_rgb = frames_rgb[:max_frames]

    session = sam3_processor.init_video_session(video=frames_rgb, inference_device=DEVICE, processing_device="cpu", video_storage_device="cpu", dtype=DTYPE)
    sam3_processor.add_text_prompt(session, prompts)

    video_info = sv.VideoInfo.from_video_path(video_path)
    sample_every_n = max(1, int(video_info.fps * sample_every_seconds))

    tracks = {}
    frame_sample= []

    for mo in sam3_model.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=len(frames_rgb) - 1):
        out = sam3_processor.postprocess_outputs(session, mo)
        frame_idx = int(mo.frame_idx)
        object_ids= out["object_ids"].tolist()
        masks= out["masks"].cpu().numpy().astype(bool)

        for i, obj_id in enumerate(object_ids):
            mask = masks[i]
            ys, xs = np.where(mask)

            if xs.size == 0:
                continue
        
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = float(mask.sum())
            cx, cy = float(xs.mean()), float(ys.mean())
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
