import cv2
import numpy as np
from .utils import get_video_duration

def animate_reveal_vertical_zoomout(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4  # seconds
    frames = int(fps * total_duration)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in range(frames):
        t = f / frames

        # -----------------------------
        # Phase 1: Vertical Reveal (0 → 0.5)
        # -----------------------------
        if t < 0.5:
            progress = t / 0.5
            eased = progress ** 2  # smooth acceleration
            reveal_h = int(height * eased)

            animated = np.zeros_like(image)
            animated[:reveal_h, :] = image[:reveal_h, :]

        # -----------------------------
        # Phase 2: Zoom Out Beyond Canvas (0.5 → 1.0)
        # -----------------------------
        else:
            progress = (t - 0.5) / 0.5
            eased = (1 - np.cos(progress * np.pi)) / 2  # smooth-out
            # zoom_factor < 1 → smaller image (zoom out)
            zoom_factor = np.interp(eased, [0, 1], [1.0, 0.3])  # goes smaller
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)
            zoomed = cv2.resize(image, (new_w, new_h))

            # Move image slightly downward & outward as it zooms out
            y_offset = int(height * 0.1 * progress)
            x1 = (width - new_w) // 2
            y1 = (height - new_h) // 2 + y_offset

            # Canvas to place zoomed image
            canvas = np.zeros_like(image)

            # If image goes outside, clip coordinates
            x1 = max(-new_w, x1)
            y1 = max(-new_h, y1)
            x2 = min(width, x1 + new_w)
            y2 = min(height, y1 + new_h)

            # Paste the visible part only
            if x1 < width and y1 < height:
                canvas[max(0, y1):y2, max(0, x1):x2] = zoomed[
                    max(0, -y1):new_h - max(0, y2 - height),
                    max(0, -x1):new_w - max(0, x2 - width)
                ]
            animated = canvas

        writer.write(animated)

    writer.release()
    return get_video_duration(out_path), frames

