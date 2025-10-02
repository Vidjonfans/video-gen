import cv2
import numpy as np
import aiohttp
import os
import uuid
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
import uvicorn







# ✅ Import animation functions
from animations import (
    animate_reveal_zoomout,
    animate_rotate_zoomin,
    animate_center_reveal_zoomout,
    fix_mp4
)













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


# ---- API endpoint ----
@app.get("/")
def home():
    return {"message": "Animation API running", "animations": ["reveal_zoomout", "rotate_zoomin","center_reveal_zoomout",]}


@app.get("/process")
async def process(
    request: Request,
    image_url: str = Query(..., description="Public image URL"),
    animation: str = Query("reveal_zoomout", description="Animation type: reveal_zoomout | rotate_zoomin | center_reveal_zoomout")
):
    img = await fetch_image(image_url)
    if img is None:
        return {"error": "Image download failed or invalid URL"}

    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")





    

    if animation == "reveal_zoomout":
        duration, frames = animate_reveal_zoomout(img, out_path)
    elif animation == "rotate_zoomin":
        duration, frames = animate_rotate_zoomin(img, out_path)
        
        elif animation == "center_reveal_zoomout":
        duration, frames = animate_center_reveal_zoomout(img, out_path)



    
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
