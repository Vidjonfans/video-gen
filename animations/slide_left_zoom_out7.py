import cv2
import numpy as np
from .utils import get_video_duration

def animate_slide_left_zoom_out7(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4  # total 4 seconds
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    written = 0
    for f in range(frames):
        t = f / frames

        # Phase 1: Slide Left (0 - 2s)
        if t < 0.5:
            phase_progress = t / 0.5  # normalize 0 → 1 for first half
            eased = 1 - (1 - phase_progress) ** 3  # ease-out cubic
            offset_x = int(eased * width * 0.5)  # move left up to half width
            M = np.float32([[1, 0, -offset_x], [0, 1, 0]])
            animated = cv2.warpAffine(image, M, (width, height))
        
        # Phase 2: Zoom Out and Zoom In (2s - 4s)
        else:
            phase_progress = (t - 0.5) / 0.5  # normalize 0 → 1 for second half
            eased = 1 - (1 - phase_progress) ** 3  # smooth
            # zoom factor oscillates: zoom out (0→1) then zoom in (1→0)
            zoom_factor = 1 + 0.6 * np.sin(eased * np.pi)  # 1 → 1.6 → 1
            new_w, new_h = int(width * zoom_factor), int(height * zoom_factor)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Center crop back to original size
            x1 = (new_w - width) // 2
            y1 = (new_h - height) // 2
            animated = resized[y1:y1+height, x1:x1+width]

        writer.write(animated)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
