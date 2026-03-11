import argparse
import shutil
import tarfile
from pathlib import Path
import requests
from extractors import extract_tar, sample_videos, organize_videos
from config import NAMESPACE, DATASET, REVISION, SOURCES, VIDEO_EXTS, K400_TRAIN_PATH_URL, REAL_PARTS


def modelscope_url(file_path):
    return f"https://www.modelscope.cn/api/v1/datasets/{NAMESPACE}/{DATASET}/repo?Source=SDK&Revision={REVISION}&FilePath={file_path}&View=False"




def download_file(file_path, out_dir):
    """
    downloading files, saving to out_dir/file_path
    skipping files which were already downloaded
    """
    out_file = out_dir / file_path
    if out_file.exists():
        return out_file

    out_dir.mkdir(parents=True, exist_ok=True)
    url = modelscope_url(file_path)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded*100/total
                    print(f"-\r{pct:.1f}% downloaded-", end="")
    return out_file


def download_kinetics400(archives_dir, n_want, skip_download):
    """
    downloading kinetics-400 train split+extracting videos from tar.gz parts
    (because one total file is too heavy so it is like more optimized)
    """
    k400_extract_dir = archives_dir / "extracted" / "Kinetics-400"
    k400_targz_dir = archives_dir / "k400_targz"

    if k400_extract_dir.exists():
        existing = sorted(p for p in k400_extract_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file())
        if len(existing) >= n_want:
            return existing
    if skip_download:
        if k400_extract_dir.exists():
            return sorted(p for p in k400_extract_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file())
        return []

    resp = requests.get(K400_TRAIN_PATH_URL, timeout=30)
    resp.raise_for_status()
    part_paths = [line.strip() for line in resp.text.splitlines() if line.strip()]
    k400_targz_dir.mkdir(parents=True, exist_ok=True)
    k400_extract_dir.mkdir(parents=True, exist_ok=True)

    all_videos = []
    if k400_extract_dir.exists():
        all_videos = sorted(p for p in k400_extract_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file())
    for part_path in part_paths:
        if len(all_videos) >= n_want:
            break

        part_name = Path(part_path).name
        part_url = part_path
        local_targz = k400_targz_dir / part_name

        if not local_targz.exists():
            try:
                with requests.get(part_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length", 0))

                    downloaded = 0
                    with open(local_targz, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded*100/total
                                print(f"-\r{pct:5.1f}% {downloaded/1024**2:,.0f}/{total/1024**2:,.0f} MB-",end="", flush=True)
            except Exception:
                if local_targz.exists():
                    local_targz.unlink()
                continue
        with tarfile.open(local_targz, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).suffix.lower() in VIDEO_EXTS]
            for m in members:
                tf.extract(m, path=k400_extract_dir)
                all_videos.append(k400_extract_dir / m.name)

    return sorted(all_videos)


def download_genvideo_real(archives_dir, skip_download):
    """
    downloading and extracting GenVideoReal multipart archive
    """
    extract_dir = archives_dir / "extracted" / "GenVideo-Real"

    if extract_dir.exists() and any(extract_dir.rglob("*.mp4")):
        return sorted(p for p in extract_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)

    if not skip_download:
        for part_name in REAL_PARTS:
            download_file(part_name, archives_dir)

    part_paths = [archives_dir / p for p in REAL_PARTS]
    missing = [p for p in part_paths if not p.exists()]
    if missing:
        return []
    combined = archives_dir / "Real_combined.tar.gz"
    if not combined.exists():
        with open(combined, "wb") as out_f:
            for pp in part_paths:
                with open(pp, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024*1024*16)
    return extract_tar(combined, extract_dir)


def process_modelscope_source(source_name, info, archives_dir, out_dir, skip_download):
    """
    downloading+extracting+sampling+organizing a single source
    """
    archive_name, label, n_want = info["archive"], info["label"], info["count"]
    archive_path = archives_dir / archive_name
    extract_dir = archives_dir / "extracted" / source_name

    if extract_dir.exists() and any(extract_dir.rglob("*.mp4")):
        videos = sorted(p for p in extract_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    else:
        if not skip_download:
            archive_path = download_file(archive_name, archives_dir)
        if not archive_path.exists():
            return
        videos = extract_tar(archive_path, extract_dir)
    sampled = sample_videos(videos, n_want)
    organize_videos(source_name, sampled, label, out_dir)


def parse_sources(source_args):
    """
    parsing arguments
    """
    if source_args is None or source_args == ["all"]:
        return {k: dict(v) for k, v in SOURCES.items()}

    selected = {}
    for arg in source_args:
        if ":" in arg:
            name, count_str = arg.split(":", 1)
            count = int(count_str)
        else:
            name = arg
            count = None
        matched = None
        for src_name in SOURCES:
            if src_name.lower() == name.lower():
                matched = src_name
                break
        if matched is None:
            print(f"-unknown source '{name}', skipping-")
            continue

        entry = dict(SOURCES[matched])
        if count is not None:
            entry["count"] = count
        selected[matched] = entry
    return selected


def main():
    parser = argparse.ArgumentParser(description="download GenVideo-100K subsets")
    parser.add_argument("--out-dir", type=str, default="./data/raw", help="output directory")
    parser.add_argument("--archives-dir", type=str, default=None, help="directory for downloaded archives")
    parser.add_argument("--skip-download", action="store_true", help="only extract and organize existing archives")
    parser.add_argument("--sources", nargs="*", default=None, help="sources to download, something like: --sources ZeroScope SVD:5000")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    archives_dir = Path(args.archives_dir).resolve() if args.archives_dir else out_dir / "archives"
    selected = parse_sources(args.sources)

    if not selected:
        return

    for source_name, info in selected.items():
        if source_name == "Kinetics-400":
            n_want = info["count"]
            all_k400 = download_kinetics400(archives_dir, n_want, args.skip_download)
            kinetics_videos = sample_videos(all_k400, n_want)
            if kinetics_videos:
                organize_videos("Kinetics-400", kinetics_videos, "real", out_dir)
            continue

        if source_name == "GenVideo-Real":
            n_want = info["count"]
            videos = download_genvideo_real(archives_dir, args.skip_download)
            if videos:
                sampled = sample_videos(videos, n_want)
                organize_videos("GenVideo-Real", sampled, "real", out_dir)
            continue

        process_modelscope_source(
            source_name, info, archives_dir, out_dir,
            args.skip_download,
        )



if __name__ == "__main__":
    main()
