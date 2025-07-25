import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def download_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
