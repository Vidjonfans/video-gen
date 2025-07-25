from flask import Flask, request, jsonify, send_file
import os
import cv2
from animation import animate_face
from utils import download_image_from_url
import uuid

app = Flask(__name__)
OUTPUT_FOLDER = "static/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/animate", methods=["POST"])
def animate():
    data = request.get_json()
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    try:
        img = download_image_from_url(image_url)
        frames = animate_face(img)
        video_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4().hex}.mp4")

        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        return jsonify({"video_url": f"/{video_path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/output/<filename>")
def serve_video(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='video/mp4')

if __name__ == "__main__":
    app.run(debug=True)
