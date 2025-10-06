import cv2
import numpy as np
from .utils import get_video_duration

def animate_rotate_zoomin(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4
    frames = fps * total_duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    center = (width // 2, height // 2)
    written = 0

    for f in range(frames):
        t = f / frames
        angle = 360 * t
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        zoom_factor = np.interp(t, [0, 1], [1.0, 1.5])
        M[0, 0] *= zoom_factor
        M[0, 1] *= zoom_factor
        M[1, 0] *= zoom_factor
        M[1, 1] *= zoom_factor
        M[0, 2] += center[0] * (1 - zoom_factor)
        M[1, 2] += center[1] * (1 - zoom_factor)

        rotated = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        writer.write(rotated)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
