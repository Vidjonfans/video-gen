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

        if t < 0.7:
            # 🌀 Phase 1: Deep Blur -> Reveal (0s - 2.8s)
            progress = t / 0.7
            eased = progress ** 2
            # Deep blur (even stronger)
            blur_strength = int(np.interp(1 - eased, [0, 1], [55, 1]))  # deep blur → sharp

            if blur_strength % 2 == 0:
                blur_strength += 1  # must be odd for cv2.GaussianBlur

            blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
            animated = blurred

        else:
            # 🎞️ Phase 2: Gentle camera shake (2.8s - 4s)
            progress = (t - 0.7) / 0.3
            eased = np.sin(progress * np.pi * 3) * (1 - progress) * 5  # decreasing oscillation
            shift_x = int(eased)
            shift_y = int(eased * 0.5)

            # translate (simulate shake)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            animated = cv2.warpAffine(image, M, (width, height))

        writer.write(animated)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
