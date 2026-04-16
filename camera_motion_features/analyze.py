import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from extractor import extract_vggt
from features import extract_features
from visualize import batch_comparison, single_video_dashboard

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def find_videos(directory):
    """get all video files from a directory"""
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file())


def run_single(args):
    """run analysis for one video"""
    path = args.video
    if not Path(path).is_file():
        print(f"Video not found: {path}")
        sys.exit(1)

    print(f"Processing {path}")
    vggt_out = extract_vggt(path, n_frames=args.n_frames, cache_dir=args.cache_dir)
    feats = extract_features(vggt_out)

    print("Feature Summary:")
    for name, value in sorted(feats.items()):
        print(f"  {name:30s}  {value:.6f}")

    if not args.no_viz:
        out = args.output or "report.png"
        single_video_dashboard(vggt_out, feats, out)
        print(f"Dashboard -> {out}")


def run_batch(args):
    """run analysis for real vs generated videos"""
    real_vids = find_videos(args.dir_real)
    gen_vids = find_videos(args.dir_generated)

    if not real_vids:
        print(f"No videos in {args.dir_real}")
        sys.exit(1)
    if not gen_vids:
        print(f"No videos in {args.dir_generated}")
        sys.exit(1)

    print(f"Found {len(real_vids)} real, {len(gen_vids)} generated videos")

    all_feats = []
    labels = []
    names = []

    for vp in tqdm(real_vids, desc="Real"):
        try:
            out = extract_vggt(str(vp), n_frames=args.n_frames, cache_dir=args.cache_dir)
            all_feats.append(extract_features(out))
            labels.append("real")
            names.append(vp.name)
        except Exception as e:
            print(f"Skip {vp.name}: {e}")

    for vp in tqdm(gen_vids, desc="Generated"):
        try:
            out = extract_vggt(str(vp), n_frames=args.n_frames, cache_dir=args.cache_dir)
            all_feats.append(extract_features(out))
            labels.append("generated")
            names.append(vp.name)
        except Exception as e:
            print(f"Skip {vp.name}: {e}")

    if not all_feats:
        print("No videos processed successfully")
        sys.exit(1)

    csv_path = args.csv or "features.csv"
    out_path = args.output or "comparison.png"

    batch_comparison(all_feats, labels, names, output_path=None if args.no_viz else out_path, csv_path=csv_path)
    print(f"CSV -> {csv_path}")

    if not args.no_viz:
        print(f"Plots -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Video Camera Physics Analyzer (VGGT)")
    p.add_argument("--video", help="Single video path")
    p.add_argument("--dir-real", help="Directory with real videos")
    p.add_argument("--dir-generated", help="Directory with generated videos")

    p.add_argument("--n-frames", type=int, default=32, help="Frames to sample (default: 32)")
    p.add_argument("--output", help="Output image path")
    p.add_argument("--csv", help="Output CSV path (batch)")
    p.add_argument("--no-viz", action="store_true", help="Skip visualisation")
    p.add_argument("--cache-dir", help="Cache directory for VGGT results")

    args = p.parse_args()

    if args.video:
        run_single(args)
    elif args.dir_real and args.dir_generated:
        run_batch(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
