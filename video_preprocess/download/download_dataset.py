import argparse
from pathlib import Path

from utils import find_videos, download, extract, sample, copy_to_dest
from download_specific_real import process_genvideo_real, process_k400
from config import SOURCES



def process_source(name, info, tmp_dir, out_dir, skip_download):
    url, label, count = info["url"], info["label"], info["count"]

    if "FilePath=" in url:
        archive_name = url.split("FilePath=")[-1].split("&")[0]
    else:
        archive_name = Path(url).name
    archive = tmp_dir / archive_name
    extracted = tmp_dir / "extracted" / name

    videos = find_videos(extracted)
    if not videos:
        if skip_download:
            if not archive.exists():
                return
        else:
            download(url, archive)

        videos = extract(archive, extracted)
        archive.unlink(missing_ok=True)

    picked = sample(videos, count)
    dest = out_dir / label / name
    copy_to_dest(picked, dest)
    print(f"{name} done -- {len(picked)} videos in {dest}")


def parse_sources(args):
    if args is None or args == ["all"]:
        return {k: dict(v) for k, v in SOURCES.items()}
    selected = {}
    for name in args:
        match = next((k for k in SOURCES if k.lower() == name.lower()), None)
        if not match:
            print(f"not found {name} -- skip")
            continue
        selected[match] = dict(SOURCES[match])
    return selected


def main():
    p = argparse.ArgumentParser(description="download video datasets")
    p.add_argument("--out-dir", default="./data/raw")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--sources", nargs="*", default=None, help="all or specific from config source")
    p.add_argument("--num-each", type=int, default=None, help="number of samples")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    tmp_dir = out_dir / "tmp"

    selected = parse_sources(args.sources)

    if args.num_each is not None:
        for info in selected.values():
            info["count"] = args.num_each

    for name, info in selected.items():
        src_type = info.get("type")
        if src_type == "multipart":
            process_genvideo_real(info, tmp_dir, out_dir, args.skip_download)
        elif src_type == "k400":
            process_k400(info, tmp_dir, out_dir, args.skip_download)
        else:
            process_source(name, info, tmp_dir, out_dir, args.skip_download)


if __name__ == "__main__":
    main()
