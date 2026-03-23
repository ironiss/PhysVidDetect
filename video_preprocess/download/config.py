SEED = 42
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

MODELSCOPE_PATH = "https://www.modelscope.cn/api/v1/datasets/cccnju/GenVideo-100K/repo?Source=SDK&Revision=master&FilePath={}&View=False"

SOURCES = {
    # fake data
    "ZeroScope":    {"url": MODELSCOPE_PATH.format("ZeroScope.tar.gz"), "label": "fake", "count": 3000},
    "SVD":  {"url": MODELSCOPE_PATH.format("SVD.tar.gz"), "label": "fake", "count": 3000},
    "VideoCrafter": {"url": MODELSCOPE_PATH.format("VideoCrafter.tar.gz"),   "label": "fake", "count": 3000},
    "Pika": {"url": MODELSCOPE_PATH.format("pika.tar.gz"), "label": "fake", "count": 3000},
    "DynamicCrafter": {"url": MODELSCOPE_PATH.format("DynamicCrafter.tar.gz"), "label": "fake", "count": 3000},
    "SD":   {"url": MODELSCOPE_PATH.format("SD.tar.gz"), "label": "fake", "count": 3000},
    "SEINE":    {"url": MODELSCOPE_PATH.format("SEINE.tar.gz"), "label": "fake", "count": 3000},
    "Latte":    {"url": MODELSCOPE_PATH.format("Latte.tar.gz"), "label": "fake", "count": 3000},
    "OpenSora": {"url": MODELSCOPE_PATH.format("OpenSora.tar.gz"), "label": "fake", "count": 3000},

    # real data
    "MSRVTT":   {"url": "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip", "label": "real", "count": 3000},
    "GenVideo-Real":    {"label": "real", "count": 3000, "type": "multipart", "parts": ["Real_part_aa", "Real_part_ab", "Real_part_ac"]},
    "Kinetics-400": {"label": "real", "count": 3000, "type": "k400"},
}

K400_TRAIN_PATH_URL = "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"
