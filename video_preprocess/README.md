# Video preprocess

Pull source datasets (real + fake), sample them down to a fixed count per source, and sync the resulting raw pool with Hugging Face. Output of this stage feeds [`segmentation/`](../segmentation/).

## Setup

Copy `.env.example` to `.env` and fill in tokens (`HF_TOKEN`, `MODELSCOPE_SDK_TOKEN`).

## download/

| File | What |
|---|---|
| `config.py` | source URLs + per-source sample counts (GenVideo fakes, MSRVTT, GenVideo-Real, K400, UCF101) |
| `download_dataset.py` | entry: download archives, extract, sample, copy to `out_dir/{real,fake}/<src>/` |
| `download_specific_real.py` | special handlers for multipart (GenVideo-Real) and Kinetics-400 |
| `utils.py` | shared download / extract / sample / copy helpers |
| `download_*_colab.ipynb` | Colab equivalents |

```
# everything from config.py
python download/download_dataset.py --out-dir ./data/raw

# subset only
python download/download_dataset.py --out-dir ./data/raw --sources Pika SVD UCF101 --num-each 1000
```

**Output:** `data/raw/{real,fake}/<source>/*.mp4`

## hf_processing/

| File | What |
|---|---|
| `upload_to_hf.py` | push a local folder to an HF dataset repo (`upload_large_folder`) |
| `download_from_hf.py` | pull the dataset back via `snapshot_download` |
| `filter_real_colab.ipynb` | manual filtering of the real pool |

```
python hf_processing/upload_to_hf.py   --folder ./data/raw --repo-id ironiss/PhysVid-Det-DATA
python hf_processing/download_from_hf.py --out-dir ./data/raw --repo-id ironiss/PhysVid-Det-DATA
```
