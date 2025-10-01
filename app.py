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

# ---- Animate: point expanding + zoom out ----
def animate_image(image, out_path, frames=40, fps=12):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    written = 0
    for f in range(frames):
        t = f / frames

        # ---- Expanding circle ----
        radius = int((min(width, height) // 4) * (t*2 if t < 0.5 else 1))
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

        animated = cv2.bitwise_and(image, mask)

        # ---- Zoom out effect at the end ----
        if t > 0.6:
            zoom_factor = 1.0 + (t - 0.6) * 1.5  # gradual zoom-out
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)
            zoomed = cv2.resize(animated, (new_w, new_h))

            # crop center to original size
            x1 = (new_w - width) // 2
            y1 = (new_h - height) // 2
            animated = zoomed[y1:y1+height, x1:x1+width]

        writer.write(animated)
        written += 1

    writer.release()

    # ✅ Duration calculate
    cap = cv2.VideoCapture(out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    duration = 0
    if fps_val > 0:
        duration = total_frames / fps_val

    return written, duration

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
    return {"message": "Point animation API running"}

@app.get("/process")
async def process(request: Request, image_url: str = Query(..., description="Public image URL")):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed or invalid URL"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")
    frame_count, duration = animate_image(img, out_path)

    # ✅ Browser friendly
    fix_mp4(out_path)

    # ✅ Full public URL
    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    return {
        "video_url": f"{base_url}/outputs/{file_name}",
        "frames_written": frame_count,
        "duration_seconds": duration
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)
