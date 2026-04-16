import argparse
import csv
import gc
import json
import os
import sys
import warnings
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import upload_folder
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ALL_FEATURE_KEYS
from combined_features import extract_all_latent_features


N_FEATURES = len(ALL_FEATURE_KEYS)
MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512
LATENT_SCALE = 0.18215
DDIM_STEPS = 25
K_FRAMES = 8
CACHE_DIR = "latent_cache"



def setup_pipeline(model_id, device):
    """load SD pipeline -- VAE Unet scheduler + empty text embeds"""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.enable_attention_slicing()
    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(DDIM_STEPS, device=device)

    tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_embeds = text_encoder(tokens.input_ids)[0]
    return vae, unet, scheduler, text_embeds, dtype



@torch.no_grad()
def encode_frames_batch(vae, frame_paths, img_size, latent_scale, device, dtype):
    """encode frames -> z0 latents (batch VAE forward)"""
    tensors = []
    for fp in frame_paths:
        pil = Image.open(fp).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
        arr = np.array(pil).astype(np.float32)/255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(x)

    batch = torch.stack(tensors).to(device, dtype=dtype)
    batch = batch*2 - 1
    z0 = vae.encode(batch).latent_dist.sample()*latent_scale
    return z0



@torch.no_grad()
def ddim_invert_batch(z0_batch, unet, scheduler, text_embeds):
    """deterministic DDIM inversion (z0 -> zT (noise space))"""
    B = z0_batch.shape[0]
    z = z0_batch.clone()
    alphas = scheduler.alphas_cumprod.to(z0_batch.device)
    ts = list(scheduler.timesteps)[::-1]
    te = text_embeds.expand(B, -1, -1)

    for i, t in enumerate(ts[:-1]):
        eps = unet(z, t, encoder_hidden_states=te).sample
        a_t = alphas[t]
        x0_hat = (z - torch.sqrt(1-a_t)*eps) / (torch.sqrt(a_t)+1e-8)
        t_next = ts[i+1]
        a_next = alphas[t_next]
        z = torch.sqrt(a_next)*x0_hat + torch.sqrt(1 - a_next)*eps
    return z



def load_masks_at_latent_res(vid_dir, frame_indices, latent_h, latent_w, mask_shape):
    """load + resize masks -- to latent resolution per frame"""
    masks_dir = os.path.join(vid_dir, "masks")
    if not os.path.isdir(masks_dir):
        return None

    obj_dirs = sorted([
        d for d in os.listdir(masks_dir)
        if os.path.isdir(os.path.join(masks_dir, d)) and d.startswith("obj_")
    ])

    if not obj_dirs:
        return None

    masks_per_frame = []
    for fi in frame_indices:
        frame_masks = {}
        for obj_d in obj_dirs:
            obj_id = int(obj_d.split("_")[1])
            mask_path = os.path.join(masks_dir, obj_d, f"{fi:05d}.npy")
            if not os.path.exists(mask_path):
                continue
            packed = np.load(mask_path)
            mask = np.unpackbits(packed)[:mask_shape[0]*mask_shape[1]].reshape(mask_shape).astype(bool)
            mask_small = cv2.resize(
                mask.astype(np.uint8), (latent_w, latent_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            if mask_small.sum()>=4:
                frame_masks[obj_id] = mask_small
        masks_per_frame.append(frame_masks)

    return masks_per_frame



def process_one(vid_dir, vae, unet, scheduler, text_embeds, device, dtype, k_frames=K_FRAMES, img_size=IMG_SIZE, batch_size=K_FRAMES, cache_dir=None):
    """full video pipeline -- frames -> z0/eps -> features"""
    vid_dir = str(vid_dir)
    meta_path = os.path.join(vid_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    frames_dir = os.path.join(vid_dir, "frames")
    if not os.path.isdir(frames_dir):
        return None

    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith((".jpeg", ".jpg", ".png"))
    ])

    if len(frame_files)<2:
        return None
    n = len(frame_files)
    if n<=k_frames:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n-1, k_frames).round().astype(int).tolist()

    frame_paths = [os.path.join(frames_dir, frame_files[i]) for i in indices]
    frame_indices = [int(Path(frame_files[i]).stem) for i in indices]
    z0_all = []
    eps_all = []

    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start:start+batch_size]
        z0_batch = encode_frames_batch(vae, batch_paths, img_size, LATENT_SCALE, device, dtype)
        zT_batch = ddim_invert_batch(z0_batch, unet, scheduler, text_embeds)
        z0_all.append(z0_batch.detach().cpu().float().numpy())
        eps_all.append(zT_batch.detach().cpu().float().numpy())

        del z0_batch, zT_batch
        torch.cuda.empty_cache()

    if not z0_all:
        return None

    z0_stack = np.concatenate(z0_all, axis=0)
    eps_stack = np.concatenate(eps_all, axis=0)

    if z0_stack.shape[0]<2:
        return None

    latent_h, latent_w = z0_stack.shape[2], z0_stack.shape[3]
    mask_shape = tuple(meta.get("mask_shape", [0, 0]))
    if mask_shape[0]>0 and mask_shape[1]>0:
        masks = load_masks_at_latent_res(vid_dir, frame_indices, latent_h, latent_w, mask_shape)
    else:
        masks = None
    if cache_dir is not None:
        vid_name = os.path.basename(vid_dir)
        cache_path = os.path.join(cache_dir, f"{vid_name}.npz")
        np.savez_compressed(cache_path, z0=z0_stack, eps=eps_stack,
                            frame_indices=np.array(frame_indices),
                            mask_shape=np.array(mask_shape))

    vec = extract_all_latent_features(z0_stack, eps_stack, masks)
    label_str = meta.get("label", "unknown")
    label = 1 if label_str == "real" else 0

    return {
        "features": vec,
        "label": label,
        "path": meta.get("video_path", os.path.basename(vid_dir)),
    }



def patch_features_from_cache(cache_dir, data_dir, out_path):
    """recompute features from cached z0/eps (cpu)"""
    cache_dir = Path(cache_dir)
    data_dir = Path(data_dir)
    cache_files = sorted(cache_dir.glob("*.npz"))
    print(f"patch mode: {len(cache_files)} cached videos in {cache_dir}")
    results = []
    failed = 0
    for cf in tqdm(cache_files, desc="recomputing features"):
        vid_name = cf.stem
        vid_dir = data_dir / vid_name
        meta_path = vid_dir / "meta.json"

        if not meta_path.exists():
            failed += 1
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            data = np.load(cf)
            z0_stack = data["z0"]
            eps_stack = data["eps"]
            frame_indices = data["frame_indices"].tolist()
            mask_shape = tuple(data["mask_shape"])
            latent_h, latent_w = z0_stack.shape[2], z0_stack.shape[3]
            if mask_shape[0]>0 and mask_shape[1]>0:
                masks = load_masks_at_latent_res(str(vid_dir), frame_indices,
                                                  latent_h, latent_w, mask_shape)
            else:
                masks = None

            vec = extract_all_latent_features(z0_stack, eps_stack, masks)

            label_str = meta.get("label", "unknown")
            label = 1 if label_str == "real" else 0

            results.append({
                "features": vec,
                "label": label,
                "path": meta.get("video_path", os.path.basename(str(vid_dir))),
            })
        except Exception as e:
            tqdm.write(f"error {vid_name}: {e}")
            failed += 1

    print(f"recomputed: {len(results)} ok, {failed} failed")

    if not results:
        print("no results")
        return

    save_results(results, out_path)


def save_results(results, out_path):
    """save feature results to H5+CSV"""
    feats_arr = np.stack([r["features"] for r in results]).astype(np.float32)
    labels_arr = np.array([r["label"] for r in results], dtype=np.int32)
    paths_list = [r["path"] for r in results]

    with h5py.File(out_path, "w") as hf:
        dt_str = h5py.special_dtype(vlen=str)
        hf.create_dataset("features", data=feats_arr, dtype="f4")
        hf.create_dataset("label", data=labels_arr, dtype="i4")
        ds_path = hf.create_dataset("path", (len(results),), dtype=dt_str)
        ds_name = hf.create_dataset("feat_names", (N_FEATURES,), dtype=dt_str)
        for i, p in enumerate(paths_list):
            ds_path[i] = p
        for i, k in enumerate(ALL_FEATURE_KEYS):
            ds_name[i] = k

    csv_path = str(Path(out_path).with_suffix(".csv"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"] + list(ALL_FEATURE_KEYS))
        writer.writeheader()
        for r in results:
            row = {"path": r["path"], "label": r["label"]}
            for k, v in zip(ALL_FEATURE_KEYS, r["features"]):
                row[k] = float(v)
            writer.writerow(row)

    valid = np.isfinite(feats_arr).all(axis=1).sum()
    print(f"saved {len(results)} rows to {out_path} + {csv_path}")
    print(f"{N_FEATURES} features, fully valid (no NaN): {valid}/{len(results)}")


def parser_vals():
    parser = argparse.ArgumentParser(description="extract latent features via DDIM inversion")
    parser.add_argument("--out", type=str, default="latent_features.h5")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--k-frames", type=int, default=K_FRAMES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--patch-features", action="store_true")
    parser.add_argument("--hf-upload-repo", type=str, default=None)
    parser.add_argument("--hf-upload-every", type=int, default=200)
    args = parser.parse_args()
    return args


def auto_upload_cache(cache_dir, hf_upload_repo, done):
    """upload cached latents to hf periodically"""
    if not hf_upload_repo or not cache_dir:
        return
    try:
        tqdm.write(f"uploading cache ({done} videos done) to {hf_upload_repo}")
        upload_folder(
            folder_path=cache_dir,
            path_in_repo="latent_cache",
            repo_id=hf_upload_repo,
            repo_type="dataset",
        )
        tqdm.write(f"cache uploaded")
    except Exception as e:
        tqdm.write(f"upload failed: {e}")


def main():
    args = parser_vals()

    if args.patch_features:
        patch_features_from_cache(args.cache_dir, args.data_dir, args.out)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading: {args.model}")

    vae, unet, scheduler, text_embeds, dtype = setup_pipeline(args.model, device)
    print(f"DDIM steps: {DDIM_STEPS}")
    print(f"batch size: {args.batch_size}")

    cache_dir = None
    if not args.no_cache:
        cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"caching z0/eps to {cache_dir}")

    data_dir = Path(args.data_dir)
    vid_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d/"meta.json").exists()])

    if args.n_shards>1:
        vid_dirs = [v for i, v in enumerate(vid_dirs) if i % args.n_shards == args.shard]
        print(f"shard {args.shard}/{args.n_shards}: {len(vid_dirs)} videos")
    else:
        print(f"found {len(vid_dirs)} videos")

    hf_upload_repo = args.hf_upload_repo
    hf_upload_every = args.hf_upload_every
    last_upload_count = 0

    results = []
    done = 0
    failed = 0

    pbar = tqdm(vid_dirs, desc="extracting")
    for vid_dir in pbar:
        try:
            row = process_one(vid_dir, vae, unet, scheduler, text_embeds, device, dtype, k_frames=args.k_frames, img_size=args.img_size, batch_size=args.batch_size, cache_dir=cache_dir)
        except Exception as e:
            tqdm.write(f"error {vid_dir.name}: {e}")
            row = None

        if row:
            results.append(row)
            done += 1
        else:
            failed += 1
        pbar.set_postfix(ok=done, fail=failed)

        if hf_upload_repo and done>0 and (done - last_upload_count)>=hf_upload_every:
            auto_upload_cache(cache_dir, hf_upload_repo, done)
            last_upload_count = done

        gc.collect()
        torch.cuda.empty_cache()

    auto_upload_cache(cache_dir, hf_upload_repo, done)

    print(f"done: {done} ok, {failed} failed")

    if not results:
        print("no results")
        return

    out_path = args.out
    if args.n_shards>1:
        out_path = str(Path(args.out).with_suffix(f".shard{args.shard}.h5"))

    save_results(results, out_path)





if __name__ == "__main__":
    main()
