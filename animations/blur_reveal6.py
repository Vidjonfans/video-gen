import cv2
import numpy as np
from .utils import get_video_duration

def animate_blur_reveal(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4
    frames = fps * total_duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    written = 0
    for f in range(frames):
        t = f / frames

        # 🌀 Smooth Blur-to-Clear transition (0 → 4s)
        progress = t  # linear time
        eased = 1 - (1 - progress) ** 3  # ease-out cubic for smoothness

        # Blur gradually decreases from 45 to 1
        blur_strength = int(np.interp(eased, [0, 1], [45, 1]))

        # Gaussian blur needs odd kernel size
        if blur_strength % 2 == 0:
            blur_strength += 1

        # Apply decreasing blur
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        animated = blurred

        writer.write(animated)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
