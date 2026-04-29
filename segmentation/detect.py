import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("av").setLevel(logging.ERROR)

import re
import av
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    hf_logging.disable_default_handler()
except Exception:
    pass


LLAVA_DEVICE = None
LLAVADTYPE = None
LLAVA_MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"

llava_model = None
llava_processor = None

LLAVA_PROMPT = (
    "Look carefully at this specific video and name only the prominent objects, "
    "animals or people that you actually see. "
    "Do NOT guess or list common things. "
    "If you only see one thing, list one. "
    "Skip background like sky, grass, walls, lighting, water, road. "
    "Answer with a short comma-separated list of nouns referring to what is in this video."
)




def init_device(gpu_id= 0):
    """set the CUDA device for the worker"""
    global LLAVA_DEVICE, LLAVADTYPE
    if torch.cuda.is_available():
        LLAVA_DEVICE = f"cuda:{gpu_id}"
        LLAVADTYPE = torch.float16
    else:
        LLAVA_DEVICE = "cpu"
        LLAVADTYPE = torch.float32



def load_llava():
    """loading lava model"""
    global llava_model, llava_processor
    if llava_model is not None:
        return
    
    llava_processor = VideoLlavaProcessor.from_pretrained(LLAVA_MODEL_ID)
    llava_model = (VideoLlavaForConditionalGeneration.from_pretrained(LLAVA_MODEL_ID, torch_dtype=LLAVADTYPE).to(LLAVA_DEVICE).eval())



def read_video_clip(video_path, num_frames= 8):
    """read evenly spaced frames from video"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames

    if total <= 0:
        total = sum(1 for _ in container.decode(video=0))
        container.seek(0)

    indices = set(np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist())
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break

    container.close()
    return np.stack(frames) if frames else np.empty((0,))



def parse_object_names(raw_answer, top_n= 5):
    """parse object names from model output"""
    raw = raw_answer.strip().rstrip(".")
    parts = re.split(r"[,\n]+", raw)

    names = []
    for part in parts:
        name = re.sub(r"^\s*[\d]+[.)]\s*", "", part)
        name = re.sub(r"^\s*[-•*]\s*", "", name)
        name = name.strip().lower()
        if not name or len(name) > 30 or len(name) < 2:
            continue
        if name.count(" ") > 3:
            continue
        if name not in names:
            names.append(name)

    return names[:top_n]



def detect_objects(video_path, top_n= 5):
    """ run Video-LLaVA and return object names"""
    load_llava()

    clip = read_video_clip(video_path, num_frames=8)
    if clip.size == 0:
        return ["object"]

    prompt = f"USER: <video>\n{LLAVA_PROMPT}\nASSISTANT:"
    inputs = llava_processor(text=prompt, videos=clip, return_tensors="pt")
    inputs = {k: v.to(LLAVA_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = llava_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
        )

        

    prompt_len = inputs["input_ids"].shape[1]
    answer = llava_processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    names = parse_object_names(answer, top_n=top_n)
    if not names:
        names = ["object"]

    return names




def reset_models():
    """clearing cached models"""
    global llava_model, llava_processor
    llava_model = None
    llava_processor = None



