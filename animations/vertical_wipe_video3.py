import cv2
import numpy as np
import requests
# आपको यहाँ 'ImageSequenceClip' का उपयोग करना चाहिए, न कि MoviePyClip का,
# क्योंकि आपने उसे इस नाम से आयात नहीं किया है।
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip 


# ✅ I. सहायक फ़ंक्शन (Helper Functions)
# ----------------------------------------------------------------------

def load_image(source):
    """Load image from numpy array, local path, or URL."""
    if isinstance(source, np.ndarray):
        return source
    if not isinstance(source, str):
        raise ValueError("❌ Invalid image source type. Must be path, URL, or numpy array.")
    
    # ... (URL/Path Loading Logic as before) ...
    if source.startswith("http://") or source.startswith("https://"):
        try:
            resp = requests.get(source, timeout=10)
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"❌ Failed to load image from URL: {e}")
    else:
        img = cv2.imread(source)

    if img is None:
        raise ValueError(f"❌ Could not load image from: {source}")
    return img


def ease_out_quart(t):
    """Smoother ease-out effect (0 to 1)."""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2


def zoom_frame(img, scale, target_size):
    """Apply centered zoom and ensures output matches target_size (W, H) exactly."""
    w, h = target_size 
    
    zoomed_w = max(1, int(w * scale))
    zoomed_h = max(1, int(h * scale))

    zoomed = cv2.resize(img, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR) 
    zh, zw = zoomed.shape[:2] 

    start_y = (zh - h) // 2
    start_x = (zw - w) // 2
    
    cropped = zoomed[
        max(0, start_y):min(zh, start_y + h), 
        max(0, start_x):min(zw, start_x + w)
    ]
    
    if cropped.shape[:2] != (h, w):
         final_frame = cv2.resize(cropped, (w, h))
    else:
         final_frame = cropped

    return final_frame

# ----------------------------------------------------------------------
# ✅ II. मुख्य एनीमेशन फ़ंक्शन (Main Animation Function)
# ----------------------------------------------------------------------

def animate_vertical_wipe_video3 (image_paths, out_path="output_vertical_wipe.mp4", fps=30):
    
    # 🚨 यहाँ मैंने आपके वीडियो उदाहरण के लिए एक न्यूनतम आवश्यकता जोड़ दी है
    if len(image_paths) < 1: 
        raise ValueError("❌ कम से कम एक छवि आवश्यक है।")

    # 📏 वीडियो सेटिंग्स
    target_size = (1080, 1920) # (Width, Height)
    W, H = target_size 
    
    # ⏱️ समय सेटिंग्स (आपके वीडियो के अनुसार)
    duration_zoom_in = 0.7      
    duration_hold = 3.0         
    duration_transition = 0.7   
    NUM_STRIPES = 7             

    # 🖼️ फ्रेम गणना
    frames_zoom = int(duration_zoom_in * fps)
    frames_hold = int(duration_hold * fps)
    frames_transition = int(duration_transition * fps)
    
    # 🔍 स्केल सेटिंग्स
    ZOOM_IN_START_SCALE = 2.05  
    ZOOM_IN_END_SCALE = 1.0
    
    frames_list = []
    
    print(f"[INFO] 🎬 वीडियो एनीमेशन शुरू हो रहा है ({len(image_paths)} छवियां, {fps} FPS)")

    for index, img_path in enumerate(image_paths):
        print(f"[INFO] Processing Image {index + 1}/{len(image_paths)}...")
        try:
            img = load_image(img_path)
            current_img_bgr = cv2.resize(img, target_size)
        except Exception as e:
            print(f"[ERROR] ❌ Image {index + 1} failed: {e}. Skipping.")
            continue

        # ----------------------------------------------------
        # B. ऊर्ध्वाधर वाइप ट्रांज़िशन (Vertical Wipe Transition)
        # ----------------------------------------------------
        if index > 0:
            
            stripe_width = W // NUM_STRIPES
            black_frame = np.zeros((H, W, 3), dtype=np.uint8) 

            for i in range(frames_transition):
                t = i / frames_transition
                wipe_progress = ease_out_quart(t)
                
                transition_frame = current_img_bgr.copy()
                
                for j in range(NUM_STRIPES):
                    start_x = j * stripe_width
                    end_x = (j + 1) * stripe_width
                    
                    wipe_height = int(H * wipe_progress)
                    
                    transition_frame[H - wipe_height:H, start_x:end_x] = black_frame[H - wipe_height:H, start_x:end_x]
                
                frames_list.append(cv2.cvtColor(transition_frame, cv2.COLOR_BGR2RGB))


        # ----------------------------------------------------
        # C. नई छवि का Zoom-In और Hold
        # ----------------------------------------------------
        
        # ✅ चरण 1: Zoom-In (0.7s)
        for i in range(frames_zoom):
            t = i / frames_zoom
            scale = ZOOM_IN_START_SCALE - (ZOOM_IN_START_SCALE - ZOOM_IN_END_SCALE) * t 
            frame = zoom_frame(current_img_bgr, scale, target_size)
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ✅ चरण 2: Hold (3.0s)
        final_zoomed_frame_rgb = cv2.cvtColor(
            zoom_frame(current_img_bgr, ZOOM_IN_END_SCALE, target_size), 
            cv2.COLOR_BGR2RGB
        )
        for _ in range(frames_hold):
            frames_list.append(final_zoomed_frame_rgb)
            
    # ✅ III. वीडियो निर्यात (Export Video)
    if not frames_list:
        raise ValueError("❌ कोई भी फ्रेम उत्पन्न नहीं हुआ।")
    
    # 🚨 त्रुटि यहाँ थी: MoviePyClip को ImageSequenceClip से बदल दिया गया है,
    # क्योंकि ImageSequenceClip को आपने कोड की शुरुआत में आयात किया था।
    clip = ImageSequenceClip(frames_list, fps=fps) 
    
    clip.write_videofile(
        out_path, 
        codec="libx264", 
        audio=False,
        logger=None,
        bitrate="5000k" 
    )
    
    # कुल अवधि की गणना
    total_duration_per_image = duration_zoom_in + duration_hold
    total_duration = (total_duration_per_image * len(image_paths)) + (duration_transition * max(0, len(image_paths) - 1))

    print(f"\n[INFO] ✅ वीडियो सफलतापूर्वक बनाया गया → {out_path} (Duration: {total_duration:.2f}s, Resolution: {W}x{H})")
    
    return total_duration, len(frames_list)