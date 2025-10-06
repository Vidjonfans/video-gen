import cv2
import numpy as np
from .utils import get_video_duration

def animate_smooth_zoom_pan(image, out_path, fps=30):
    height, width = image.shape[:2]
    total_duration = 4.5  # matches your video (≈4.47s)
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    center = (width // 2, height // 2)
    written = 0

    for f in range(frames):
        t = f / frames
        # ease-in-out curve for smooth motion
        ease = 3 * t**2 - 2 * t**3

        # subtle rotation (0° → 12°)
        angle = np.interp(ease, [0, 1], [0, 12])
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # soft zoom-in (1.0 → 1.25)
        zoom_factor = np.interp(ease, [0, 1], [1.0, 1.25])
        M[0, 0] *= zoom_factor
        M[0, 1] *= zoom_factor
        M[1, 0] *= zoom_factor
        M[1, 1] *= zoom_factor

        # slight upward pan (simulate motion like your video)
        dy = np.interp(ease, [0, 1], [0, -height * 0.05])
        M[0, 2] += center[0] * (1 - zoom_factor)
        M[1, 2] += center[1] * (1 - zoom_factor) + dy

        # apply transformation
        transformed = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        writer.write(transformed)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
