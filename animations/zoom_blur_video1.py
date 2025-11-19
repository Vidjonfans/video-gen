import cv2
import numpy as np
import requests
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


# ✅ I. सहायक फ़ंक्शन (Helper Functions)
#Required Imgae 3

def load_image(source):
    """Load image from numpy array, local path, or URL."""
    # ... (Implementation is correct and robust) ...
    if isinstance(source, np.ndarray):
        return source
    if not isinstance(source, str):
        raise ValueError("❌ Invalid image source type. Must be path, URL, or numpy array.")

    if source.startswith("http://") or source.startswith("https://"):
        try:
            resp = requests.get(source, timeout=10)
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"❌ Failed to load image from URL: {e}")
    else:
        img = cv2.imread(source)

    if img is None:
        raise ValueError(f"❌ Could not load image from: {source}")
    return img


def ease_in_out(t):
    """Smooth ease-in-out curve."""
    return t * t * (3 - 2 * t)


def zoom_frame(img, scale, target_size):
    """
    Apply centered zoom to image and ensures output matches target_size EXACTLY (W, H).
    """
    w, h = target_size # target_size is (Width, Height)
    
    # 1. Zoom the image
    zoomed_w = max(1, int(w * scale))
    zoomed_h = max(1, int(h * scale))

    zoomed = cv2.resize(img, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)
    
    zh, zw = zoomed.shape[:2] # Zoomed height, Zoomed width

    # 2. Center Crop to (h, w)
    start_y = (zh - h) // 2
    start_x = (zw - w) // 2
    end_y = start_y + h
    end_x = start_x + w

    cropped = zoomed[
        max(0, start_y):min(zh, end_y), 
        max(0, start_x):min(zw, end_x)
    ]
    
    # 3. Final Forced Resize (Guarantees MoviePy compatibility)
    final_frame = cv2.resize(cropped, (w, h))
    
    return final_frame


# ✅ II. मुख्य एनीमेशन फ़ंक्शन (Main Animation Function)

def animate_zoom_blur_video1(image_paths, out_path="output.mp4", fps=30):
    """
    Animate images with 0.3s Zoom-In transition followed by 3.0s Hold.
    """

    # 🔥 FIX #1 — Remove nested lists (Render bug)
    clean_images = []
    for img in image_paths:
        if isinstance(img, list):   # Render returns [[array]]
            img = img[0]
        clean_images.append(img)

    image_paths = clean_images

    if len(image_paths) < 1:
        raise ValueError("❌ कम से कम एक छवि आवश्यक है।")

    target_size = (1080, 1920)

    # 🔥 FIX #2 — First image no longer uses load_image()
    first_img = image_paths[0]
    if first_img is None or not isinstance(first_img, np.ndarray):
        raise ValueError("❌ पहली छवि अवैध है।")

    first_img = cv2.resize(first_img, target_size)

    frames_list = []
    
    duration_zoom = 0.9
    duration_hold = 3.0
    total_frames_zoom = int(duration_zoom * fps)
    total_frames_hold = int(duration_hold * fps)

    START_SCALE = 1.2
    END_SCALE = 1.0
    SCALE_DIFF = START_SCALE - END_SCALE

    print(f"[INFO] 🎬 एनीमेशन शुरू हो रहा है ({len(image_paths)} छवियां, {fps} FPS)")

    for img in image_paths:

        # 🔥 FIX #3 — Remove load_image completely
        if img is None or not isinstance(img, np.ndarray):
            print("[ERROR] Invalid image array. Skipping.")
            continue

        img = cv2.resize(img, target_size)

        # Zoom frames
        for i in range(total_frames_zoom):
            t = ease_in_out(i / total_frames_zoom)
            scale = START_SCALE - SCALE_DIFF * t
            frame = zoom_frame(img, scale, target_size)
            frames_list.append(frame)

        # Hold frames
        for _ in range(total_frames_hold):
            frames_list.append(img.copy())

    if not frames_list:
        raise ValueError("❌ कोई फ्रेम उत्पन्न नहीं हुआ।")

    expected_shape = (target_size[1], target_size[0], 3)
    if not all(f.shape == expected_shape for f in frames_list):
        raise ValueError("❌ Final Frame Size Mismatch.")

    clip = ImageSequenceClip(
        [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_list],
        fps=fps
    )

    clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)

    duration = (duration_zoom + duration_hold) * len(image_paths)
    print(f"[INFO] ✔ वीडियो बनाया गया → {out_path}")

    return duration, len(frames_list)
