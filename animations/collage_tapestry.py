import cv2
import numpy as np
import requests
import math
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


# ✅ Background image (fixed)
BACKGROUND_URL = "https://res.cloudinary.com/dvsubaggj/image/upload/v1761447077/Screenshot_2025-10-19_155811_rkg3nz.png"

def load_image_from_url(url):
    """Download image from URL and return OpenCV image."""
    try:
        resp = requests.get(url, timeout=10)
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Could not load image: {e}")
        return None

def ease_in_out(t):
    """Smooth ease-in-out interpolation (0→1)."""
    return t * t * (3 - 2 * t)

# NOTE: Since you are using moviepy in your final run_animation_sync, 
# you should ideally use moviepy to write the video here too, 
# or ensure the CV2 video writer is finalized correctly and fix_mp4 is applied.
# I am assuming moviepy is imported via utils or globally for consistency.

# Function to return video duration (Kept as is for compatibility)
def get_video_duration(path):
    """Return video duration in seconds."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps

# CORRECTED FUNCTION SIGNATURE to accept an image list
def animate_collage_tapestry(images: list, out_path, fps=24):
    """
    Animate images over a total duration with zoom + fade effects.
    - Each image gets an equal slice of the total duration.
    """
    if len(images) < 1:
         raise ValueError("Image list cannot be empty for collage_tapestry.")
    
    bg_img = load_image_from_url(BACKGROUND_URL)
    if bg_img is None:
        raise ValueError("Failed to load background image.")

    bg_h, bg_w = bg_img.shape[:2]
    num_images = len(images)
    
    # Total duration is dynamic based on number of images
    duration_per_image = 3.0 # seconds per image
    total_duration = duration_per_image * num_images
    frames = int(fps * total_duration)

    # Resize all images to fit nicely
    img_size = (int(bg_w * 0.6), int(bg_h * 0.6))
    resized_images = [cv2.resize(img, img_size) for img in images]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (bg_w, bg_h))

    for f in range(frames):
        t = f / fps
        frame = bg_img.copy()

        # Determine which image is active (e.g., if t=3.5s and duration_per_image=3.0)
        active_index = math.floor(t / duration_per_image)
        
        # Handle the transition point for the last image
        if active_index >= num_images:
            active_index = num_images - 1

        img = resized_images[active_index]
        
        # Local time for the active image (0.0 to 1.0)
        local_t = (t % duration_per_image) / duration_per_image 

        # Zoom effect (scale 1 → 1.2)
        scale = 1 + 0.2 * ease_in_out(local_t)
        h, w = img.shape[:2]
        zoomed = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Center the zoomed image
        zh, zw = zoomed.shape[:2]
        x = bg_w // 2 - zw // 2
        y = bg_h // 2 - zh // 2

        # Fade effect: Fade in at the start, peak at middle, fade out at the end
        # 0.0 → 1.0 → 0.0 over the image duration
        # alpha = ease_in_out(1 - abs(0.5 - local_t) * 2) # Original logic was slightly buggy for continuous loop
        
        # New fade logic: Fade in for first 0.5s, max alpha, fade out for last 0.5s
        fade_duration = 0.5 / duration_per_image # 0.5 seconds fade duration in local_t units
        
        if local_t < fade_duration:
            # Fade in
            alpha = local_t / fade_duration
        elif local_t > 1.0 - fade_duration:
            # Fade out
            alpha = (1.0 - local_t) / fade_duration
        else:
            # Full visibility
            alpha = 1.0
            
        alpha = ease_in_out(np.clip(alpha, 0.0, 1.0)) # Apply smooth curve and clip

        roi = frame[y:y+zh, x:x+zw]
        
        # Ensure ROI matches zoomed size for blending
        if roi.shape[:2] == zoomed.shape[:2]:
            blended = cv2.addWeighted(roi, 1 - alpha, zoomed, alpha, 0)
            frame[y:y+zh, x:x+zw] = blended

        writer.write(frame)

    writer.release()
    print(f"[INFO] Collage tapestry video created → {out_path}")
    return get_video_duration(out_path), frames