import cv2
import numpy as np
from .utils import get_video_duration

def animate_smooth_zoom_pan(image, out_path, fps=30):
    height, width = image.shape[:2]
    total_duration = 4.5
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    center = (width // 2, height // 2)
    written = 0

    # Create a bokeh-like blurred version for depth-of-field blending
    blurred_bg = cv2.GaussianBlur(image, (35, 35), 25)

    # Precompute a mask for depth effect (center sharp, edges blurred)
    mask = np.zeros((height, width), np.float32)
    cv2.circle(mask, center, int(min(height, width) * 0.45), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (151, 151), 50)

    for f in range(frames):
        t = f / frames
        ease = 3 * t**2 - 2 * t**3

        # ✅ Slow Zoom-in
        zoom_factor = np.interp(ease, [0, 1], [1.0, 1.25])

        # ✅ Subtle rotation & live photo motion
        angle = np.interp(ease, [0, 1], [0, 10])
        M = cv2.getRotationMatrix2D(center, angle, zoom_factor)

        # ✅ Slight upward pan (cinematic drift)
        dy = np.interp(ease, [0, 1], [0, -height * 0.05])
        dx = np.sin(t * np.pi * 2) * 3  # tiny left-right drift
        M[0, 2] += center[0] * (1 - zoom_factor) + dx
        M[1, 2] += center[1] * (1 - zoom_factor) + dy

        # Transform the frame
        frame = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)

        # ✅ Bokeh / Depth-of-Field blending
        frame = frame.astype(np.float32) / 255.0
        bg = blurred_bg.astype(np.float32) / 255.0
        blended = frame * mask[..., None] + bg * (1 - mask[..., None])

        # ✅ Dreamy / Soft lighting (glow)
        glow = cv2.GaussianBlur(blended, (0, 0), 5)
        dreamy = cv2.addWeighted(blended, 0.85, glow, 0.35, 0)

        # ✅ Color grading (soft contrast + warmth)
        dreamy = np.clip(dreamy * 1.05 + 0.02, 0, 1)
        dreamy[..., 2] = np.clip(dreamy[..., 2] * 1.05, 0, 1)  # warmer tone
        dreamy = (dreamy * 255).astype(np.uint8)

        writer.write(dreamy)
        written += 1

    writer.release()
    return get_video_duration(out_path), written
