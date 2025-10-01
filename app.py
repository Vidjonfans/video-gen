import cv2
import numpy as np
import aiohttp
import os
import uuid
import subprocess
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
import uvicorn

# FastAPI app
app = FastAPI()

# Static serve for outputs folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# ---- Helper: download image ----
async def fetch_image(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                nparr = np.frombuffer(data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] fetch_image failed: {e}")
        return None


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


# ---- API endpoint ----
@app.get("/")
def home():
    return {"message": "Animation API running", "animations": ["reveal_zoomout", "rotate_zoomin"]}


@app.get("/process")
async def process(
    request: Request,
    image_url: str = Query(..., description="Public image URL"),
    animation: str = Query("reveal_zoomout", description="Animation type: reveal_zoomout | rotate_zoomin")
):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed or invalid URL"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")

    if animation == "reveal_zoomout":
        duration, frames = animate_reveal_zoomout(img, out_path)
    elif animation == "rotate_zoomin":
        duration, frames = animate_rotate_zoomin(img, out_path)
    else:
        return {"error": "Invalid animation type!"}

    # ✅ Browser friendly
    fix_mp4(out_path)

    # ✅ Public URL
    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    return {
        "video_url": f"{base_url}/outputs/{file_name}",
        "frames_written": frames,
        "duration_seconds": duration,
        "animation": animation
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)
