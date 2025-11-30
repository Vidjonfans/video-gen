import cv2
import numpy as np
import aiohttp
import os
import uuid
import asyncio
import requests  # 🔹 Added for Cloudinary upload
from fastapi import FastAPI, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys, os
sys.path.append(os.path.join(os.getcwd(), "venv", "Lib", "site-packages"))
print("✅ Custom site-packages path added:", os.path.join(os.getcwd(), "venv", "Lib", "site-packages"))

# ✅ Import animations + utils
from animations.collage_tapestry import animate_collage_tapestry
from animations.slide_left_zoom_out7 import animate_slide_left_zoom_out7
from animations.zoom_blur_video1 import animate_zoom_blur_video1
from animations.zoom_transition2 import animate_zoom_transition2
from animations.vertical_wipe_video3 import animate_vertical_wipe_video3
from animations.center_reveal_zoomout import animate_three_slide_right_to_left






from animations.utils import fix_mp4, add_audio_to_video


# ✅ FastAPI app
app = FastAPI(
    title="🎬 Image Animation API with Audio",
    description="Generate animated videos from images using cinematic effects and custom audio.",
    version="2.3.0"
)

# ✅ Enable CORS for all origins (Flutter Web fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Output folder setup
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")


# ---- Health check ----
@app.head("/")
async def head_check():
    return Response(status_code=200)


# ---- Root endpoint ----
@app.get("/")
async def home():
    return {
        "message": "🎥 Animation API is running!",
        "available_animations": [
            "collage_tapestry",
            "zoom_blur_video1",
            "slide_left_zoom_out7"
            "zoom_transition2",
            "vertical_wipe_video3",
            "three_slide_right_to_left"
            
        ],
        "example_request": "/process?image_url=https://yourimage.jpg&animation=zoomin_zoomout_fadein2&audio_url=https://youraudio.aac"
    }


# ---- Helper: Download image ----
async def fetch_image(url: str):
    """Download image from public URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Invalid image URL: {url}")
                    return None
                data = await resp.read()
                nparr = np.frombuffer(data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] fetch_image failed: {e}")
        return None


# ---- Helper: Upload video to Cloudinary ----
def upload_to_cloudinary(local_path: str):
    """Upload the video to Cloudinary and return its secure URL."""
    cloud_name = "dvsubaggj"
    upload_preset = "flutter_unsigned_upload"
    url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"

    try:
        with open(local_path, "rb") as file_data:
            res = requests.post(
                url,
                files={"file": file_data},
                data={"upload_preset": upload_preset},
                timeout=120
            )
        result = res.json()
        if res.status_code == 200:
            secure_url = result["secure_url"]
            print(f"[✅] Cloudinary upload successful → {secure_url}")
            return secure_url
        else:
            print(f"[❌] Cloudinary upload failed → {result}")
            return None
    except Exception as e:
        print(f"[ERROR] Upload to Cloudinary failed: {e}")
        return None

# ---- Animation runner (Updated to accept 'imgs' list) ----
# NOTE: The parameter name is changed from 'img' to 'imgs' for clarity, 
# and it now expects a list of images.






def run_animation_sync(imgs: list, out_path, animation, audio_url=None):
    """Run selected animation with a list of images and optionally add audio."""
    try:
        # ✅ Select animation
        if animation == "collage_tapestry":
            duration, frames = animate_collage_tapestry(imgs, out_path) 
        elif animation == "slide_left_zoom_out7":
            duration, frames = animate_slide_left_zoom_out7(imgs, out_path) 
        elif animation == "zoom_blur_video1":
            duration, frames = animate_zoom_blur_video1(imgs, out_path)
        elif animation == "zoom_transition2":
            duration, frames = animate_zoom_transition2(imgs, out_path)
        elif animation == "vertical_wipe_video3":
            duration, frames = animate_vertical_wipe_video3(imgs, out_path)
        elif animation == "three_slide_right_to_left":
            duration, frames = animate_three_slide_right_to_left(imgs, out_path)




        else:
            raise ValueError(f"Invalid animation type: {animation}")
        
        # ... (rest of the run_animation_sync function remains the same) ...
        # ... (fix_mp4, add_audio_to_video logic) ...
        
        # ⚠️ WARNING: Your provided code for run_animation_sync was incomplete/incorrectly nested.
        # Assuming the rest of the logic (fix_mp4, add_audio_to_video) is outside the if/else block
        
        # ✅ Re-encode for browser
        fix_mp4(out_path)

        # ✅ Add custom audio (if provided)
        if audio_url:
            out_with_audio = out_path.replace(".mp4", "_audio.mp4")
            added = add_audio_to_video(out_path, audio_url, out_with_audio)
            if added:
                os.replace(out_with_audio, out_path)
                print(f"[INFO] Audio added from {audio_url}")

        print(f"[INFO] Animation '{animation}' completed successfully → {out_path}")
        return duration, frames

    except Exception as e:
        print(f"[ERROR] Animation failed: {e}")
        raise

# ---- Main endpoint (Corrected to handle one batch process) ----
@app.get("/process")
async def process(
    request: Request,
    total_images: int = Query(..., ge=1, le=7, description="Total number of images user wants to upload (1–7)"),
    image_url1: str = Query(..., description="Image URL 1 (Required)"),
    image_url2: str = Query(None, description="Image URL 2"),
    image_url3: str = Query(None, description="Image URL 3"),
    image_url4: str = Query(None, description="Image URL 4"),
    image_url5: str = Query(None, description="Image URL 5"),
    image_url6: str = Query(None, description="Image URL 6"),
    image_url7: str = Query(None, description="Image URL 7"),
    animation: str = Query(..., description="Animation type (Required)"),
    audio_url: str = Query(None, description="Optional audio URL (MP3, AAC, etc.)")
):
    """
    🎞️ Accepts up to 7 image URLs, downloads them, and creates a single animated video.
    """

    print("\n\n==============================")
    print("🔥 NEW /process REQUEST ARRIVED")
    print("==============================")

    # --------------------------
    # Validate count
    # --------------------------
    if total_images < 1 or total_images > 7:
        return {"error": "❌ 'total_images' must be between 1 and 7."}

    all_urls = [image_url1, image_url2, image_url3, image_url4, image_url5, image_url6, image_url7]
    image_urls_to_use = [url for url in all_urls[:total_images] if url]

    if len(image_urls_to_use) != total_images:
        return {
            "error": f"⚠️ You specified {total_images} images but provided {len(image_urls_to_use)} URLs.",
            "hint": "Please match the number of URLs with 'total_images'."
        }

    print(f"[INFO] Downloading {total_images} image(s)...")

    # --------------------------
    # Download Images
    # --------------------------
    download_tasks = [fetch_image(url) for url in image_urls_to_use]
    images = await asyncio.gather(*download_tasks)

    # Debug line (REQUIRED)
    print("\n[DEBUG] Raw images from fetch:", [type(i) for i in images])

    # --------------------------
    # Fix nested lists (Render bug)
    # --------------------------
    clean_images = []
    for img in images:
        if isinstance(img, list):       # fix for [[array]]
            print("⚠ FIXED — nested list detected")
            img = img[0]
        clean_images.append(img)

    images = clean_images

    print("[DEBUG] After cleaning:", [type(i) for i in images])

    if any(img is None for img in images):
        return {"error": "❌ Failed to fetch one or more images from the provided URLs."}

    # --------------------------
    # Generate output file path
    # --------------------------
    out_path = os.path.join(OUTDIR, f"anim_{uuid.uuid4().hex}.mp4")

    print(f"[INFO] Running animation → {animation}")

    # --------------------------
    # Run animation safely
    # --------------------------
    try:
        loop = asyncio.get_event_loop()
        duration, frames = await loop.run_in_executor(
            None, lambda: run_animation_sync(images, out_path, animation, audio_url)
        )
    except Exception as e:
        print("\n[ERROR] Animation failed:", e)
        return {"error": f"❌ Animation processing failed: {str(e)}"}

    # --------------------------
    # Validate output existence
    # --------------------------
    if not os.path.exists(out_path):
        return {"error": "⚠ Video generation failed or missing."}

    # --------------------------
    # Upload to Cloudinary
    # --------------------------
    cloudinary_url = upload_to_cloudinary(out_path)

    if not cloudinary_url:
        return {"error": "❌ Failed to upload video to Cloudinary."}

    # --------------------------
    # Cleanup
    # --------------------------
    try:
        os.remove(out_path)
        print("[INFO] Local file removed.")
    except:
        pass

    print("[SUCCESS] Video uploaded:", cloudinary_url)

    return {
        "status": "✅ Success",
        "animation": animation,
        "image_count_used": len(images),
        "audio_attached": bool(audio_url),
        "duration_seconds": duration,
        "frames_written": frames,
        "video_url": cloudinary_url
    }






# ---- Startup Event ----
@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing Animation API...")
    await asyncio.sleep(3)
    print("✅ Ready to process requests.")


# ---- Run locally ----
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=False)
