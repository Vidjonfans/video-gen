import cv2
import numpy as np
import requests


# ---------------------------
# Helper: Load image
# ---------------------------
def load_image(source):
    if isinstance(source, np.ndarray):
        return source

    if not isinstance(source, str):
        raise ValueError("Invalid image source.")

    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=10)
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(source)

    if img is None:
        raise ValueError(f"Failed to load image: {source}")

    return img


# ---------------------------
# Easing
# ---------------------------
def ease_in_out(t):
    return t * t * (3 - 2 * t)


# ---------------------------
# Centered Zoom
# ---------------------------
def zoom_frame(img, scale, target_size):
    w, h = target_size
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    zoomed = cv2.resize(img, (new_w, new_h))

    zh, zw = zoomed.shape[:2]
    start_y = (zh - h) // 2
    start_x = (zw - w) // 2

    cropped = zoomed[
        max(0, start_y):min(zh, start_y + h),
        max(0, start_x):min(zw, start_x + w)
    ]

    final_frame = cv2.resize(cropped, (w, h))

    return final_frame



# =========================================================
# MAIN FUNCTION (No MoviePy, 100% CV2-Based)
# =========================================================
def animate_zoom_blur_video1(image_paths, out_path="output.mp4", fps=30):
    
    if len(image_paths) < 1:
        raise ValueError("At least 1 image required")

    target_size = (1080, 1920)  # (W, H)

    # Load first image
    first_img = load_image(image_paths[0])
    first_img = cv2.resize(first_img, target_size)

    duration_zoom = 0.9
    duration_hold = 3.0

    frames_zoom = int(duration_zoom * fps)
    frames_hold = int(duration_hold * fps)

    START = 1.2
    END   = 1.0
    DIFF  = START - END

    # CV2 Writer
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        target_size
    )

    total_written = 0

    for img_path in image_paths:
        img = load_image(img_path)
        img = cv2.resize(img, target_size)

        # ---------------------------
        # 1. Zoom In (0.9 sec)
        # ---------------------------
        for i in range(frames_zoom):
            t = ease_in_out(i / frames_zoom)
            scale = START - DIFF * t
            frame = zoom_frame(img, scale, target_size)
            writer.write(frame)
            total_written += 1

        # ---------------------------
        # 2. Hold (3 sec)
        # ---------------------------
        for _ in range(frames_hold):
            writer.write(img)
            total_written += 1

    writer.release()

    total_duration = (duration_zoom + duration_hold) * len(image_paths)

    print(f"[INFO] Video created: {out_path}, Duration {total_duration:.2f}s, Frames {total_written}")

    return total_duration, total_written
