import random
import shutil
import tarfile
import zipfile
from config import VIDEO_EXTS, SEED
from pathlib import Path



def extract_tar(archive_path, extract_dir):
    """
    extracting data from archives -- .tar.gz
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).suffix.lower() in VIDEO_EXTS]
        for m in members:
            tf.extract(m, path=extract_dir)
            videos.append(extract_dir / m.name)
    return sorted(videos)

def extract_zip(archive_path, extract_dir, subfolder_filter=None):
    """
    extracting data from archives -- .zip
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [m for m in zf.namelist() if Path(m).suffix.lower() in VIDEO_EXTS]
        if subfolder_filter:
            members = [m for m in members if subfolder_filter.lower() in m.lower()]
        for m in members:
            zf.extract(m, path=extract_dir)
            videos.append(extract_dir / m)
    return sorted(videos)


def sample_videos(videos, n):
    """
    choosing random videos (n of them -- from input)
    """
    if len(videos) <= n:
        return videos
    rng = random.Random(SEED)
    return sorted(rng.sample(videos, n))


def organize_videos(source_name, videos, label, final_dir):
    """
    structuring data
    """
    dest = final_dir / label / source_name
    dest.mkdir(parents=True, exist_ok=True)
    organized = []
    for v in videos:
        dst = dest / v.name
        if not dst.exists():
            shutil.copy2(v, dst)
        organized.append(dst)
    return organized


