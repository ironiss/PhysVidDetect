SEED = 42

NAMESPACE = "cccnju"
DATASET = "GenVideo-100K"
REVISION = "master"

REAL_PARTS = ["Real_part_aa", "Real_part_ab", "Real_part_ac"]
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# list of files taken here: https://github.com/chenhaoxing/DeMamba?tab=readme-ov-file
SOURCES = {
    # fake data
    "ZeroScope": {"archive": "ZeroScope.tar.gz", "label": "fake", "count": 2000},
    "SVD": {"archive": "SVD.tar.gz", "label": "fake", "count": 2000},
    "VideoCrafter": {"archive": "VideoCrafter.tar.gz", "label": "fake", "count": 2000},
    "Pika": {"archive": "pika.tar.gz", "label": "fake", "count": 2000},
    "DynamicCrafter": {"archive": "DynamicCrafter.tar.gz", "label": "fake", "count": 2000},
    "SD": {"archive": "SD.tar.gz", "label": "fake", "count": 2000},
    "SEINE": {"archive": "SEINE.tar.gz", "label": "fake", "count": 2000},
    "Latte": {"archive": "Latte.tar.gz", "label": "fake", "count": 2000},
    "OpenSora": {"archive": "OpenSora.tar.gz", "label": "fake", "count": 2000},

    # real data
    "GenVideo-Real": {"archive": "multipart", "label": "real", "count": 3000},
    "Kinetics-400": {"archive": "k400", "label": "real", "count": 3000},
}

K400_TRAIN_PATH_URL = "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"
K400_VAL_PATH_URL = "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt"
K400_BASE_URL = "https://s3.amazonaws.com/kinetics/400"
