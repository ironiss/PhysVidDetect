import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from huggingface_hub import login, upload_large_folder


load_dotenv(Path(__file__).resolve().parent / ".env")

def main():
    parser = argparse.ArgumentParser(description="upload data to hf")
    parser.add_argument("--folder", type=str, default="../data/raw", required=True,help="local folder to move to the hf")
    parser.add_argument("--repo-id", type=str, default="ironiss/PhysVidDetect-v1", help="HF dataset repo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        login(token=token)

    folder = Path(args.folder).resolve()
    upload_large_folder(folder_path=str(folder),repo_id=args.repo_id, repo_type="dataset")

if __name__ == "__main__":
    main()
    print(f"-uploaded-")