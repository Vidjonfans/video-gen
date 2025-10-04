import cv2
import numpy as np
import subprocess
import os

# ============================================
# ---- Animation 1: Vertical Reveal + Zoom out (4 sec)
# ============================================
def animate_reveal_zoomout(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4  # duration = 4 sec
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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

    writer.release()
    return get_video_duration(out_path)


# ============================================
# ---- Animation 2: Rotate + Zoom in (6 sec)
# ============================================
def animate_rotate_zoomin(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 6  # duration = 6 sec
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    center = (width // 2, height // 2)

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

    writer.release()
    return get_video_duration(out_path)


# ============================================
# ---- Animation 3: Center Reveal + Zoom out (5 sec)
# ============================================
def animate_center_reveal_zoomout(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 5  # duration = 5 sec
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in range(frames):
        t = f / frames
        if t < 0.5:
            progress = t / 0.5
            eased = progress ** 2
            reveal_h = int(height * eased * 0.5)
            animated = np.zeros_like(image)
            center_y = height // 2
            y1 = max(center_y - reveal_h, 0)
            y2 = min(center_y + reveal_h, height)
            animated[y1:y2, :] = image[y1:y2, :]
        else:
            progress = (t - 0.5) / 0.5
            eased = (1 - np.cos(progress * np.pi)) / 2
            zoom_factor = np.interp(eased, [0, 1], [1.0, 1.5])
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)
            zoomed = cv2.resize(image, (new_w, new_h))
            x1 = (new_w - width) // 2
            y1 = (new_h - height) // 2
            animated = zoomed[y1:y1+height, x1:x1+width]

        writer.write(animated)

    writer.release()
    return get_video_duration(out_path)


# ============================================
# ---- Animation 4: Smooth Pan + Zoom (7 sec)
# ============================================
def animate_pan_zoom(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 7  # duration = 7 sec
    frames = int(fps * total_duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in range(frames):
        t = f / frames
        zoom_factor = np.interp(np.sin(t * np.pi), [-1, 1], [0.9, 1.1])
        pan_x = int((width * 0.1) * np.sin(t * 2 * np.pi))

        new_w = int(width * zoom_factor)
        new_h = int(height * zoom_factor)
        zoomed = cv2.resize(image, (new_w, new_h))

        x1 = (new_w - width) // 2 + pan_x
        y1 = (new_h - height) // 2
        x1 = np.clip(x1, 0, new_w - width)
        y1 = np.clip(y1, 0, new_h - height)
        animated = zoomed[y1:y1+height, x1:x1+width]

        writer.write(animated)

    writer.release()
    return get_video_duration(out_path)


# ============================================
# ---- Helper: calculate duration ----
# ============================================
def get_video_duration(out_path):
    cap = cv2.VideoCapture(out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames / fps_val if fps_val > 0 else 0


# ============================================
# ---- Browser-friendly fix (ffmpeg re-encode)
# ============================================
def fix_mp4(out_path):
    fixed_path = out_path.replace(".mp4", "_fixed.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                fixed_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        os.replace(fixed_path, out_path)
        print("[INFO] MP4 re-encoded for browser compatibility")
    except Exception as e:
        print("[ERROR] ffmpeg failed:", e)
