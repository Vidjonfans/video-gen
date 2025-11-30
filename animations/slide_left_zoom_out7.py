import cv2
import numpy as np
from .utils import get_video_duration


def animate_slide_left_zoom_out7(images, out_path, fps=24):
    """
    UPDATED:
    - Accepts 3 images in list
    - Each image: 3 sec (right → left slide)
    - After slide: 1.5 sec zoom effect
    - Total per-image = 4.5 sec
    """

    if not isinstance(images, list) or len(images) < 1:
        raise ValueError("❌ 'images' must be a list of 3 images.")

    # Resize all to match first image
    height, width = images[0].shape[:2]
    images = [cv2.resize(img, (width, height)) for img in images]

    seconds_slide = 3       # right → left slide
    seconds_zoom  = 1.5     # zoom animation
    frames_slide  = int(fps * seconds_slide)
    frames_zoom   = int(fps * seconds_zoom)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    written = 0

    # ---------------------------------------------------
    # LOOP FOR ALL 3 IMAGES
    # ---------------------------------------------------
    for img in images:

        # ===============================================
        # 1) RIGHT → LEFT SLIDE (3 sec)
        # ===============================================
        for f in range(frames_slide):
            t = f / frames_slide  # 0 → 1

            # right → left
            start_x = width
            end_x = 0
            offset_x = int(start_x + (end_x - start_x) * t)

            frame = np.zeros_like(img)

            x1 = max(offset_x, 0)
            x2 = min(offset_x + width, width)

            img_x1 = max(0, -offset_x)
            img_x2 = img_x1 + (x2 - x1)

            if x2 > x1:
                frame[:, x1:x2] = img[:, img_x1:img_x2]

            writer.write(frame)
            written += 1

        # ===============================================
        # 2) ZOOM ANIMATION (1.5 sec)
        # ===============================================
        for f in range(frames_zoom):
            t = f / frames_zoom   # 0 → 1

            eased = 1 - (1 - t) ** 3  # smooth ease-out

            zoom_factor = 1 + 0.6 * np.sin(eased * np.pi)  # 1 → 1.6 → 1
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)

            resized = cv2.resize(img, (new_w, new_h))

            x1 = (new_w - width) // 2
            y1 = (new_h - height) // 2

            frame = resized[y1:y1 + height, x1:x1 + width]

            writer.write(frame)
            written += 1

    writer.release()

    total_duration = ((seconds_slide + seconds_zoom) * len(images))
    print(f"[INFO] Video ready → Duration {total_duration:.2f}s, Frames {written}")

    return get_video_duration(out_path), written
