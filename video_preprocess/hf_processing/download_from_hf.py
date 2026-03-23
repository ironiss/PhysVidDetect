import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login,snapshot_download

load_dotenv(Path(__file__).resolve().parent / ".env")

def main():
    parser = argparse.ArgumentParser(description="getting previously downloaded data to HF here locally")
    parser.add_argument("--out-dir", type=str, default="../data/raw",  help="directory to download into")
    parser.add_argument("--repo-id", type=str, default="ironiss/PhysVidDetect-v1", help="HF dataset repo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        login(token=token)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id=args.repo_id,repo_type="dataset",local_dir=str(out_dir),local_dir_use_symlinks=False)

if __name__ == "__main__":
    main()
    print(f"-downloaded-")
