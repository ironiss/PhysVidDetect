import sys
import os
import argparse
import gc
import hashlib
import json
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path
import cv2
import numpy as np
import torch
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT / "segmentation"))
sys.path.insert(0, str(PROJ_ROOT / "latent_features"))
sys.path.insert(0, str(PROJ_ROOT / "physics_features"))
sys.path.insert(0, str(PROJ_ROOT / "camera_motion_features"))

import detect
import video_tracker as vt
import extractor as cam
from preprocess import preprocess_video, TARGET_FPS, MAX_SHORT_SIDE
from detect import detect_objects, init_device
from masks import save_tracks, load_tracks
from video_tracker import sam3_text_tracks_from_video, hf_login
from extract_latent_features import setup_pipeline, process_one, MODEL_ID, IMG_SIZE, K_FRAMES
from config import ALL_FEATURE_KEYS
from physics_features import extract_all_physics_features, PHYSICS_FEATURE_KEYS
from extractor import extract_vggt
from features import extract_features as extract_camera



SAVED_MODELS = PROJ_ROOT / "saved_models"





def free_gpu(label=""):
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"--gpu after {label}] free={free_b/1024**3:.1f} / total={total_b/1024**3:.1f} GiB--")



def unload_segmentation_models():
    if getattr(detect, "llava_model", None) is not None:
        detect.llava_model.to("cpu")

    detect.llava_model = None
    detect.llava_processor = None

    if getattr(vt, "sam3_model", None) is not None:
        vt.sam3_model.to("cpu")
    
    vt.sam3_model = None
    vt.sam3_processor = None

    free_gpu("segmentation unload")


def unload_latent_models(refs):
    for k in list(refs.keys()):
        obj = refs[k]
    
        if hasattr(obj, "to"):
            obj.to("cpu")
    
        refs[k] = None
    
    free_gpu("latent unload")


def unload_camera_models():
    if getattr(cam, "cached_model", None) is not None:
        cam.cached_model.to("cpu")
    
    cam.cached_model = None
    cam.cached_device = None
    cam.cached_dtype = None
    free_gpu("camera unload")



def run_segmentation(video_path, out_dir):
    """Video-LLaVA + SAM 3 -> masks + meta.json into out_dir/<hash>/"""
    
    print("[1/4] segmentation -> Video-LLaVA + SAM 3")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_hash = hashlib.md5(str(video_path.resolve()).encode()).hexdigest()[:12]
    vid_out = out_dir / vid_hash
    frames_dir = vid_out / "frames"

    if (vid_out / "meta.json").exists():
        print(f"--skip (already segmented): {vid_out}")
    
        return vid_out

    hf_login()
    init_device(0)

    frame_paths, fps, resolution = preprocess_video(str(video_path), str(frames_dir), target_fps=TARGET_FPS, max_short_side=MAX_SHORT_SIDE)
    
    if len(frame_paths) < 5:
        raise RuntimeError(f"video too short -- only {len(frame_paths)} frames")
    
    print(f"--preprocessed: {len(frame_paths)} frames, {fps:.0f}fps, {resolution}")

    prompts = detect_objects(str(video_path), top_n=5)
    print(f"--Video-LLaVA prompts: {prompts}")

    frames_rgb = []
    for p in frame_paths:
        bgr = cv2.imread(p)
        if bgr is not None:
            frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    
    frames_rgb = np.stack(frames_rgb)

    tracks, _ = sam3_text_tracks_from_video(str(video_path), prompts=prompts, frames_rgb=frames_rgb)
    
    if not tracks:
        raise RuntimeError("SAM 3 found no objects to track")
    
    print(f"--SAM 3 tracked {len(tracks)} objects")

    save_tracks(tracks, prompts, str(video_path), label="unknown", fps=fps, resolution=resolution, frame_paths=frame_paths, out_dir=str(vid_out))
    
    return vid_out



def extract_latent_features(vid_dir):
    """SD VAE + DDIM inversion -> latent dict"""
    print("[2/4] latent features -> SD VAE + DDIM inversion")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet, scheduler, text_embeds, dtype = setup_pipeline(MODEL_ID, device)
    refs = {"vae": vae, "unet": unet, "text_embeds": text_embeds}

    try:
        res = process_one(str(vid_dir), vae, unet, scheduler, text_embeds, device, dtype, k_frames=K_FRAMES, img_size=IMG_SIZE, batch_size=K_FRAMES, cache_dir=None)
    finally:
        unload_latent_models(refs)

    if res is None:
        raise RuntimeError("latent feature extraction failed")
    
    return dict(zip(ALL_FEATURE_KEYS, res["features"].astype(float)))



def extract_physics_features(vid_dir):
    """physics dict from saved masks"""
    print("[3/4] physics features -> from object masks")

    tracks, frame_paths, _ = load_tracks(str(vid_dir))
    if not tracks or not frame_paths:
        raise RuntimeError("no tracks to compute physics features from")
    
    vec = extract_all_physics_features(tracks, frame_paths)
    return dict(zip(PHYSICS_FEATURE_KEYS, vec.astype(float)))



def extract_camera_features(video_path):
    """VGGT -> camera dict"""
    print("[4/4] camera features -> VGGT")
    
    try:
        vggt_out = extract_vggt(str(video_path), n_frames=32, cache_dir=None)
        feats = extract_camera(vggt_out)
    finally:
        unload_camera_models()
    
    return feats



def load_classifier():
    model_path = SAVED_MODELS / "final_model.json"
    scaler_path = SAVED_MODELS / "final_scaler.pkl"
    names_path = SAVED_MODELS / "feature_names.json"
    meta_path = SAVED_MODELS / "final_metadata.json"

    for p in (model_path, scaler_path, names_path):
    
        if not p.exists():
            raise FileNotFoundError(f"missing pretrained artifact: {p}")

    clf = XGBClassifier()
    clf.load_model(str(model_path))

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)

    metadata = {}
    
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return clf, scaler, feature_names, metadata


def assemble_feature_vector(all_feats, feature_names, scaler):
    """Pick features in classifier order, fill NaN with train mean (post-scale = 0)."""
    missing = [n for n in feature_names if n not in all_feats]
    
    if missing:
        print(f"--WARNING: {len(missing)} features missing; substituting train mean. ")

    vec = np.array([all_feats.get(n, np.nan) for n in feature_names], dtype=np.float32)

    train_means = np.asarray(scaler.mean_, dtype=np.float32)
    nan_mask = ~np.isfinite(vec)
    
    if nan_mask.any():
        vec = np.where(nan_mask, train_means, vec)

    X_scaled = scaler.transform(vec.reshape(1, -1))
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0).astype(np.float32)
    
    return X_scaled



def predict(video_path, work_dir, skip_segmentation=False, segmented_dir=None):
    if skip_segmentation:
    
        if segmented_dir is None or not (segmented_dir / "meta.json").exists():
            raise FileNotFoundError("with --skip-segmentation you must pass --segmented-dir <DIR>")
        vid_dir = segmented_dir
    else:
        vid_dir = run_segmentation(video_path, work_dir / "segmented")
        unload_segmentation_models()

    latent_feats = extract_latent_features(vid_dir)
    physics_feats = extract_physics_features(vid_dir)
    camera_feats = extract_camera_features(video_path)

    all_feats = {**latent_feats, **physics_feats, **camera_feats}

    print("classifier -> XGBoost (saved_models/final_model.json)")
    clf, scaler, feature_names, metadata = load_classifier()
    X_scaled = assemble_feature_vector(all_feats, feature_names, scaler)

    n_dropped = len(latent_feats) + len(physics_feats) + len(camera_feats) - len(feature_names)
    print(f"--filtering: {len(feature_names)} kept (of {len(all_feats)} computed; {n_dropped} dropped at training time)")

    prob_real = float(clf.predict_proba(X_scaled)[0, 1])
    prob_fake = 1.0 - prob_real

    if prob_real >= 0.5:
        label = "real"
    else:
        label = "fake"

    return {
        "video": str(video_path),
        "prediction": label,
        "confidence": max(prob_real, prob_fake),
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "n_features_used": len(feature_names),
        "training_overall_auc": metadata.get("overall", {}).get("auc"),
        "training_overall_acc": metadata.get("overall", {}).get("acc"),
        "all_features": all_feats,
    }




def main():
    ap = argparse.ArgumentParser(description="Predict fake/real for a single video.")
    ap.add_argument("video", help="path to the input video")
    ap.add_argument("--work-dir", default=None,
                    help="dir for intermediate frames/masks (default: tmp dir, deleted on exit)")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--skip-segmentation", action="store_true",
                    help="reuse a pre-segmented directory instead of running SAM 3 again")
    ap.add_argument("--segmented-dir", default=None,
                    help="path to dir with meta.json + frames + masks (with --skip-segmentation)")
    ap.add_argument("--out-json", default=None,
                    help="if set, write the full result here")
    args = ap.parse_args()



    video_path = Path(args.video).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="dipl_predict_"))
        cleanup = not args.keep_temp

    try:
        result = predict(video_path, work_dir, skip_segmentation=args.skip_segmentation, segmented_dir=Path(args.segmented_dir).resolve() if args.segmented_dir else None)
    finally:
        if cleanup and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


    print()
    print("-" * 60)
    print(f"  video       : {result['video']}")
    print(f"  prediction  : {result['prediction'].upper()}")
    print(f"  confidence  : {result['confidence']:.3f}")
    print(f"  P(real)     : {result['prob_real']:.4f}")
    print(f"  P(fake)     : {result['prob_fake']:.4f}")
    print(f"  features    : {result['n_features_used']}")
    
    
    if result["training_overall_auc"] is not None:
        print(f"--(model trained at AUC={result['training_overall_auc']:.3f}, Acc={result['training_overall_acc']:.3f})")
    print("-" * 60)

    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"--full result -> {out}")





if __name__ == "__main__":
    main()


