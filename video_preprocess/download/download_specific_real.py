import shutil
import tarfile
from pathlib import Path
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


from utils import find_videos, download, extract, sample, copy_to_dest
from config import VIDEO_EXTS, MODELSCOPE_PATH, K400_TRAIN_PATH_URL


def process_k400(info, tmp_dir, out_dir, skip_download):
    label, count = info["label"], info["count"]
    extracted = tmp_dir / "extracted" / "Kinetics-400"
    parts_dir = tmp_dir / "k400_parts"

    videos = find_videos(extracted)
    if len(videos) >= count:
        picked = sample(videos, count)
        dest = out_dir / label / "Kinetics-400"
        copy_to_dest(picked, dest)

        print(f"Kinetics-400 done -- {len(picked)} videos in {dest}")
        return

    if skip_download:
        print(f"Kinetics-400 not enough videos, skipping (--skip-download)")
        return

    resp = requests.get(K400_TRAIN_PATH_URL, timeout=40, verify=False)
    resp.raise_for_status()
    partial_urls = [l.strip() for l in resp.text.splitlines() if l.strip()]

    parts_dir.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    for url in partial_urls:
        if len(videos) >= count:
            break
        fname = Path(url).name
        local = parts_dir / fname

        if not local.exists():
            try:
                download(url, local)
            except Exception:
                local.unlink(missing_ok=True)
                continue

        with tarfile.open(local, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and Path(m.name).suffix.lower() in VIDEO_EXTS:
                    tf.extract(m, path=extracted)
                    videos.append(extracted / m.name)
        local.unlink(missing_ok=True)

    picked = sample(videos, count)
    dest = out_dir / label / "Kinetics-400"
    copy_to_dest(picked, dest)

    print(f"Kinetics-400 done -- {len(picked)} videos in {dest}")


def process_genvideo_real(info, tmp_dir, out_dir, skip_download):
    parts = info["parts"]
    label, count = info["label"], info["count"]
    extracted = tmp_dir / "extracted" / "GenVideo-Real"

    videos = find_videos(extracted)
    if not videos:
        if not skip_download:
            for part in parts:
                download(MODELSCOPE_PATH.format(part), tmp_dir / part)

        part_files = [tmp_dir / p for p in parts]
        if any(not f.exists() for f in part_files):
            return

        combined = tmp_dir / "Real_combined.tar.gz"
        if not combined.exists():
            with open(combined, "wb") as out:
                for pf in part_files:
                    with open(pf, "rb") as inp:
                        shutil.copyfileobj(inp, out, length=1024*1024*16)

        videos = extract(combined, extracted)

        combined.unlink(missing_ok=True)
        for partials in part_files:
            partials.unlink(missing_ok=True)

    picked = sample(videos, count)
    dest = out_dir / label / "GenVideo-Real"
    copy_to_dest(picked, dest)
    
    print(f"GenVideo-Real done -- {len(picked)} videos in {dest}")
