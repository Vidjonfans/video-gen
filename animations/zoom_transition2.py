import cv2
import numpy as np
import requests
# moviepy.editor.ImageSequenceClip का उपयोग करने के लिए इसे बदल दिया गया है, 
# लेकिन आपकी वर्तमान आयात पंक्ति (import line) को बनाए रखा गया है।
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip 

# ✅ I. सहायक फ़ंक्शन (Helper Functions)
# image require 4

# (load_image, ease_out_quart, zoom_frame, blur_and_zoom_out फ़ंक्शन अपरिवर्तित रहेंगे)

def load_image(source):
    """Load image from numpy array, local path, or URL."""
    if isinstance(source, np.ndarray):
        return source
    if not isinstance(source, str):
        raise ValueError("❌ Invalid image source type. Must be path, URL, or numpy array.")

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
    return 1 - pow(1 - t, 4)

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

def blur_and_zoom_out(img, t, max_blur, start_scale, end_scale, target_size):
    """
    एक फ्रेम पर ब्लर और ज़ूम-आउट दोनों प्रभाव लागू करता है।
    t: 0.0 (शून्य ब्लर) से 1.0 (अधिकतम ब्लर) तक।
    """
    # 1. ब्लर लागू करें: t=0 पर 0 ब्लर, t=1 पर max_blur
    blur_amount = int(max_blur * t)
    # ब्लर कर्नेल साइज़ हमेशा विषम (odd) होना चाहिए
    blur_ksize = max(1, blur_amount * 2 + 1)
    
    blurred_img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    
    # 2. ज़ूम लागू करें: start_scale से end_scale तक
    # ध्यान दें: t=0 पर scale=start_scale और t=1 पर scale=end_scale
    scale = start_scale + (end_scale - start_scale) * t 
    
    # 3. ज़ूम और क्रॉप करें
    final_frame = zoom_frame(blurred_img, scale, target_size)
    
    return final_frame

# ----------------------------------------------------------------------
# ✅ II. मुख्य एनीमेशन फ़ंक्शन (Main Animation Function)
# ----------------------------------------------------------------------

def animate_zoom_transition2(image_paths, out_path="output_popout_reel.mp4", fps=30):
    """
    छवियों को ब्लर-पॉपआउट ट्रांज़िशन के साथ 1080x1920 रील प्रारूप में एनिमेट करता है।
    """
    if len(image_paths) < 1:
        raise ValueError("❌ कम से कम एक छवि आवश्यक है।")

    # 📏 वीडियो सेटिंग्स
    target_size = (1080, 1920) # (Width, Height)
    W, H = target_size 
    
    # ⏱️ समय सेटिंग्स
    duration_zoom_in = 0.9      # प्रत्येक नई छवि के लिए ज़ूम-इन (अपरिवर्तित)
    duration_hold = 3.0         # होल्ड के लिए (अपरिवर्तित)
    duration_transition = 1   # 🚨 बदला गया: ब्लर और ज़ूम-आउट ट्रांज़िशन के लिए 0.5 सेकंड
    
    # 🖼️ फ्रेम गणना
    frames_zoom = int(duration_zoom_in * fps)
    frames_hold = int(duration_hold * fps)
    frames_transition = int(duration_transition * fps) # 🚨 बदला गया
    
    # 🔍 स्केल सेटिंग्स
    ZOOM_IN_START_SCALE = 1.2
    ZOOM_IN_END_SCALE = 1.0
    
    # ट्रांज़िशन के लिए ज़ूम-आउट (पॉप-आउट इफ़ेक्ट)
    POP_OUT_START_SCALE = 1.0   # फुल-स्क्रीन से शुरू
    POP_OUT_END_SCALE = 0.8     # थोड़ा सिकुड़ जाता है
    MAX_BLUR = 15               # ब्लर को 10 से 15 तक बढ़ाया ताकि छोटे समय में अधिक इफ़ेक्ट दिखे

    frames_list = []
    last_frame_bgr = None 

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
        # B. ट्रांज़िशन (Transition) - पिछली छवि से वर्तमान छवि तक (0.5s)
        # ----------------------------------------------------
        if last_frame_bgr is not None and index > 0:
            for i in range(frames_transition):
                # t: 0.0 (शुरुआत) से 1.0 (अंत) तक
                t = ease_out_quart(i / frames_transition) 
                
                # पिछली छवि को धुंधला करें और छोटा करें (पॉप-आउट इफ़ेक्ट)
                # t=0 पर स्केल 1.0 और t=1 पर स्केल 0.8 होगा।
                out_frame = blur_and_zoom_out(
                    last_frame_bgr, 
                    t, 
                    MAX_BLUR, 
                    POP_OUT_START_SCALE, 
                    POP_OUT_END_SCALE, 
                    target_size
                )
                
                # फ्रेम को सूची में जोड़ें
                frames_list.append(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))


        # ----------------------------------------------------
        # C. नई छवि का Zoom-In और Hold
        # ----------------------------------------------------
        
        # ✅ चरण 1: Zoom-In (0.9s)
        for i in range(frames_zoom):
            t = ease_out_quart(i / frames_zoom)
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
            
        last_frame_bgr = current_img_bgr.copy()


    # ✅ III. वीडियो निर्यात (Export Video)
    if not frames_list:
        raise ValueError("❌ कोई भी फ्रेम उत्पन्न नहीं हुआ।")
    
    clip = ImageSequenceClip(frames_list, fps=fps)
    
    clip.write_videofile(
        out_path, 
        codec="libx264", 
        audio=False,
        logger=None,
        bitrate="5000k" 
    )
    
    # कुल अवधि की गणना: (Zoom + Hold) * N + (Transition * (N-1))
    total_duration_per_image = duration_zoom_in + duration_hold
    total_duration = (total_duration_per_image * len(image_paths)) + (duration_transition * max(0, len(image_paths) - 1))

    print(f"\n[INFO] ✅ वीडियो सफलतापूर्वक बनाया गया → {out_path} (Duration: {total_duration:.2f}s, Resolution: {W}x{H})")
    
    return total_duration, len(frames_list)