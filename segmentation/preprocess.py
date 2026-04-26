import os
import cv2

TARGET_FPS = 25
MAX_SHORT_SIDE = 720

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

def preprocess_video(video_path, frames_dir, target_fps= TARGET_FPS, max_short_side= MAX_SHORT_SIDE, max_frames= None):
    """decode video+normalize fps+resolution+save JPEG frames"""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    short_side = min(src_w, src_h)
    if short_side > max_short_side:
        scale = max_short_side / short_side
        new_w = int(src_w * scale) // 2 * 2  
        new_h = int(src_h * scale) // 2 * 2
    else:
        new_w, new_h = src_w, src_h

    if src_fps > 0 and target_fps > 0:
        frame_step = max(1, round(src_fps / target_fps))
    else:
        frame_step = 1

    os.makedirs(frames_dir, exist_ok=True)
    paths = []
    src_idx = 0
    out_idx = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if src_idx % frame_step == 0:
            if new_w != src_w or new_h != src_h:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            p = os.path.join(frames_dir, f"{out_idx:05d}.jpeg")
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(p)
            out_idx += 1

            if max_frames is not None and out_idx >= max_frames:
                break

        src_idx += 1

    cap.release()
    actual_fps = target_fps if frame_step > 1 else src_fps
    return paths, actual_fps, (new_w, new_h)
