import cv2
import numpy as np
import aiohttp
import os
import uuid
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
import uvicorn

# ✅ Import animation functions
from animations.reveal_zoomout import animate_reveal_zoomout
from animations.rotate_zoomin import animate_rotate_zoomin
from animations.center_reveal_zoomout import animate_center_reveal_zoomout
from animations.blur_zooming_roatation import animate_smooth_zoom_pan
from animations.reveal_vertical import animate_reveal_vertical_zoomout
from animations.blur_reveal6 import animate_blur_reveal  # 👈 your blur reveal animation

from animations.utils import fix_mp4


# ✅ FastAPI app
app = FastAPI(
    title="Image Animation API 🎞️",
    description="Generate animated videos from images using various cinematic effects.",
    version="1.0.0"
)

# ✅ Static serve for outputs folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")


# ---- Helper: download image ----
async def fetch_image(url: str):
    """Download image from a public URL and return as OpenCV array."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Invalid image URL: {url}")
                    return None
                data = await resp.read()
                nparr = np.frombuffer(data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] fetch_image failed: {e}")
        return None


# ---- Root endpoint ----
@app.get("/")
def home():
    """Root endpoint listing available animation types."""
    return {
        "message": "🎬 Animation API running successfully!",
        "available_animations": [
            "reveal_zoomout",
            "rotate_zoomin",
            "center_reveal_zoomout",
            "smooth_zoom_pan",
            "reveal_vertical_zoomout",
            "blur_reveal"  # 👈 newly added animation
        ],
        "example_request": "/process?image_url=https://yourimage.jpg&animation=blur_reveal"
    }


# ---- Main processing endpoint ----
@app.get("/process")
async def process(
    request: Request,
    image_url: str = Query(..., description="Public image URL"),
    animation: str = Query(
        "reveal_zoomout",
        description=(
            "Animation type: reveal_zoomout | rotate_zoomin | smooth_zoom_pan | "
            "reveal_vertical_zoomout | center_reveal_zoomout | blur_reveal"
        )
    ),
):
    """Download image and apply selected animation."""
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "❌ Image download failed or invalid URL"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")

    # ✅ Animation selection logic
    try:
        if animation == "reveal_zoomout":
            duration, frames = animate_reveal_zoomout(img, out_path)
        elif animation == "rotate_zoomin":
            duration, frames = animate_rotate_zoomin(img, out_path)
        elif animation == "center_reveal_zoomout":
            duration, frames = animate_center_reveal_zoomout(img, out_path)
        elif animation == "smooth_zoom_pan":
            duration, frames = animate_smooth_zoom_pan(img, out_path)
        elif animation == "reveal_vertical_zoomout":
            duration, frames = animate_reveal_vertical_zoomout(img, out_path)
        elif animation == "blur_reveal":
            duration, frames = animate_blur_reveal(img, out_path)
        else:
            return {"error": f"❌ Invalid animation type: {animation}"}

    except Exception as e:
        print(f"[ERROR] Animation failed: {e}")
        return {"error": f"Animation processing failed: {e}"}

    # ✅ Convert to browser-compatible MP4
    fix_mp4(out_path)

    # ✅ Build public URL for output video
    base_url = str(request.base_url).rstrip("/")
    file_name = os.path.basename(out_path)
    video_url = f"{base_url}/outputs/{file_name}"

    return {
        "status": "✅ Success",
        "animation": animation,
        "duration_seconds": duration,
        "frames_written": frames,
        "video_url": video_url,
    }


# ---- Run the app ----
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)
