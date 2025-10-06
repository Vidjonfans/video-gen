import cv2
import numpy as np
from .utils import get_video_duration

def animate_reveal_zoomout(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4
    frames = fps * total_duration
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    written = 0
    for f in range(frames):
        t = f / frames
        if t < 0.5:
            progress = t / 0.5
            eased = progress ** 2
            reveal_h = int(height * eased * 0.5)
            animated = np.zeros_like(image)
            animated[:reveal_h, :] = image[:reveal_h, :]
            animated[height-reveal_h:, :] = image[height-reveal_h:, :]
        else:
            progress = (t - 0.5) / 0.5
            eased = (1 - np.cos(progress * np.pi)) / 2
            zoom_factor = np.interp(eased, [0, 1], [1.0, 0.6])
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)
            zoomed = cv2.resize(image, (new_w, new_h))
            canvas = np.zeros_like(image)
            x1 = (width - new_w) // 2
            y1 = (height - new_h) // 2
            canvas[y1:y1+new_h, x1:x1+new_w] = zoomed
            animated = canvas

        writer.write(animated)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
