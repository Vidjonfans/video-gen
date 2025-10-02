import cv2
import numpy as np
import subprocess
import os

# ============================================
# ---- Animation 1: Vertical Reveal + Zoom out
# ============================================
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


# ============================================
# ---- Animation 2: Rotate (gol gol) + Zoom in
# ============================================
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
        # rotation angle (0 → 360 degrees)
        angle = 360 * t
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Zoom-in effect towards end
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



# ============================================
# ---- Animation 3: Center Reveal + Zoom out
# ============================================
def animate_center_reveal_zoomout(image, out_path, fps=24):
    height, width = image.shape[:2]
    total_duration = 4
    frames = fps * total_duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    written = 0
    for f in range(frames):
        t = f / frames
        if t < 0.5:
            # Phase 1: Center reveal (top & bottom)
            progress = t / 0.5
            eased = progress ** 2
            reveal_h = int(height * eased * 0.5)

            animated = np.zeros_like(image)
            center_y = height // 2

            # reveal top half upwards
            y1 = center_y - reveal_h
            if y1 < 0: y1 = 0
            animated[y1:center_y, :] = image[y1:center_y, :]

            # reveal bottom half downwards
            y2 = center_y + reveal_h
            if y2 > height: y2 = height
            animated[center_y:y2, :] = image[center_y:y2, :]

        else:
            # Phase 2: Zoom OUTWARDS (bahar ki taraf)
            progress = (t - 0.5) / 0.5
            eased = (1 - np.cos(progress * np.pi)) / 2  
            
            # 👇 yaha pe change kiya (1.0 → 1.5)
            zoom_factor = np.interp(eased, [0, 1], [1.0, 1.5])
            
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)
            zoomed = cv2.resize(image, (new_w, new_h))

            canvas = np.zeros_like(image)
            
            # center crop from zoomed image (overflow crop karna hoga)
            x1 = (new_w - width) // 2
            y1 = (new_h - height) // 2
            x2 = x1 + width
            y2 = y1 + height

            animated = zoomed[y1:y2, x1:x2]

        writer.write(animated)
        written += 1

    writer.release()
    return get_video_duration(out_path), written




# ---- Helper: calculate duration ----
def get_video_duration(out_path):
    cap = cv2.VideoCapture(out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = 0
    if fps_val > 0:
        duration = total_frames / fps_val
    return duration


# ---- Browser-friendly fix (ffmpeg re-encode) ----
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
        os.replace(fixed_path, out_path)  # overwrite original
        print("[INFO] MP4 re-encoded for browser compatibility")
    except Exception as e:
        print("[ERROR] ffmpeg failed:", e)
