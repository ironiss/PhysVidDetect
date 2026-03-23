import random
import shutil
import tarfile
import zipfile
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
from tqdm import tqdm

from config import VIDEO_EXTS, SEED


def find_videos(directory):
    """collecting video -- returning paths"""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def download(url, path_to_save):
    """downloading video, skipping if exists -- returning path to downloaded file"""
    if path_to_save.exists():
        return path_to_save

    path_to_save.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(path_to_save, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path_to_save.name) as bar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
    return path_to_save


def extract(archive, to_dir):
    to_dir.mkdir(parents=True, exist_ok=True)
    videos = []

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            for name in zf.namelist():
                if Path(name).suffix.lower() in VIDEO_EXTS:
                    zf.extract(name, path=to_dir)
                    videos.append(to_dir / name)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and Path(m.name).suffix.lower() in VIDEO_EXTS:
                    tf.extract(m, path=to_dir)
                    videos.append(to_dir / m.name)

    return sorted(videos)


def sample(videos, samples):
    """sampling videos based on config count"""
    if len(videos) <= samples:
        return videos
    return sorted(random.Random(SEED).sample(videos, samples))


def copy_to_dest(videos, destination):
    """copying videos to destination"""
    destination.mkdir(parents=True, exist_ok=True)
    for vid in videos:
        final = destination / vid.name
        
        if not final.exists():
            shutil.copy2(vid, final)


