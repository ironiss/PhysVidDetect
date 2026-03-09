import os
import tarfile
import tempfile
import concurrent.futures
from pathlib import Path
from urllib.request import urlopen

import requests
from tqdm import tqdm
from huggingface_hub import list_repo_files, hf_hub_download

TRAINING_ROOT = Path("TRAINING_DATA")
REAL_DIR = TRAINING_ROOT / "REAL"
FAKE_DIR = TRAINING_ROOT / "FAKE"

LIST_URL = "https://dl.fbaipublicfiles.com/video_similarity_challenge/46ef53734a4/vsc_url_list.txt"
REF_LIST = "ref_file_paths.txt"
MAX_WORKERS = 32
TIMEOUT = 30

VIDPROM_REPO = "WenhaoWang/VidProM"

import random

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")

def list_videos(root: Path):
    vids = []
    for ext in VIDEO_EXTS:
        vids += list(root.rglob(f"*{ext}"))
    return vids

def make_subset_file(video_root: Path, out_txt: Path, n: int, seed: int = 0):
    vids = list_videos(video_root)
    if len(vids) < n:
        raise ValueError(f"Requested {n} videos, but found only {len(vids)} in {video_root}")
    rng = random.Random(seed)
    subset = rng.sample(vids, n)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(str(p.resolve()) for p in subset) + "\n", encoding="utf-8")
    print(f"Wrote subset list with {n} videos → {out_txt.resolve()}")
    return subset

def download_file(url, dest_root):
    filename = Path(url).name
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_path = dest_root / filename

    if dest_path.exists():
        return False

    try:
        resp = requests.get(url, stream=True, timeout=TIMEOUT)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        tqdm.write(f"Failed: {filename} ({e})")
        return False

def download_real():
    print("\n=== DOWNLOADING REAL VIDEOS ===")

    with open(REF_LIST, "r", encoding="utf-8") as f:
        wanted_names = {Path(line.strip()).name for line in f if line.strip()}

    with urlopen(LIST_URL, timeout=60) as resp:
        url_list = [line.decode().strip() for line in resp if line.strip()]

    matching_urls = [u for u in url_list if Path(u).name in wanted_names]
    matching_urls = matching_urls[:2000]
    print(f"Found {len(matching_urls)} matching REAL files to download.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(
            tqdm(
                ex.map(lambda u: download_file(u, REAL_DIR), matching_urls),
                total=len(matching_urls),
                desc="Downloading REAL",
                unit="file",
            )
        )

    success = sum(results)
    print(f"Downloaded {success}/{len(matching_urls)} REAL files into {REAL_DIR.resolve()}")

def extract_tar(tar_path, out_root):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=out_root)
    print(f"Extracted {tar_path.name} into {out_root}")

def download_fake():
    print("\n=== DOWNLOADING FAKE VIDEOS (VidProM pika_videos) ===")

    # List all tar files without downloading anything
    all_repo_files = list(list_repo_files(VIDPROM_REPO, repo_type="dataset"))
    pika_tars = sorted([f for f in all_repo_files if f.startswith("pika_videos/") and f.endswith(".tar")])
    print(f"Found {len(pika_tars)} tar files in pika_videos. Downloading one at a time until 2000 videos collected.")

    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for tar_name in pika_tars:
            if len(list_videos(FAKE_DIR)) >= 2000:
                break

            print(f"Downloading {tar_name} ...")
            local_tar = hf_hub_download(
                repo_id=VIDPROM_REPO,
                repo_type="dataset",
                filename=tar_name,
                local_dir=tmp_path,
                local_dir_use_symlinks=False,
            )
            extract_tar(Path(local_tar), FAKE_DIR)
            Path(local_tar).unlink()  # delete tar immediately to save disk space
            print(f"  → {len(list_videos(FAKE_DIR))} fake videos so far")

    # Trim to exactly 2000
    all_fake = list_videos(FAKE_DIR)
    for extra in all_fake[2000:]:
        extra.unlink()

    print(f"Done: {len(list_videos(FAKE_DIR))} fake videos in {FAKE_DIR.resolve()}")


def main():
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    download_real()
    download_fake()
    print("\nAll downloads complete.")
    print(f"REAL → {REAL_DIR.resolve()}")
    print(f"FAKE → {FAKE_DIR.resolve()}")


if __name__ == "__main__":
    main()
