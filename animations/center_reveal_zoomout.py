import cv2
import numpy as np
import requests
from .utils import get_video_duration


def load_image_from_url(url):
    resp = requests.get(url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def animate_three_slide_right_to_left(image_urls, out_path, fps=24):
    """
    3 images → Each 3 seconds → Slide from RIGHT → LEFT.
    Total video duration = 3 * 3 = 9 seconds.
    """

    if len(image_urls) != 3:
        raise ValueError("❌ Exactly 3 images required.")

    # Load images
    images = [load_image_from_url(u) for u in image_urls]

    # All images resize to first image resolution
    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]

    seconds_per_image = 3
    frames_per_image = fps * seconds_per_image
    total_frames = frames_per_image * 3

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    written = 0

    # --------------- MAIN LOOP --------------------
    for idx, img in enumerate(images):

        for f in range(frames_per_image):
            t = f / frames_per_image   # 0 → 1 for each image

            # Start from Right → move to Center
            # Start X = +width
            # End X = 0
            start_x = w
            end_x = 0
            current_x = int(start_x + (end_x - start_x) * t)

            # Prepare blank frame
            frame = np.zeros_like(img)

            # Compute overlay area inside frame bounds
            x1 = max(current_x, 0)
            x2 = min(current_x + w, w)

            img_x1 = max(0, -current_x)
            img_x2 = img_x1 + (x2 - x1)

            # Paste image
            if x2 > x1 and img_x2 > img_x1:
                frame[:, x1:x2] = img[:, img_x1:img_x2]

            writer.write(frame)
            written += 1

    writer.release()

    return get_video_duration(out_path), written
