from flask import Flask, request, jsonify
import google.generativeai as genai
import logging

app = Flask(__name__)

# ✅ Enable logging
logging.basicConfig(level=logging.DEBUG)

# ✅ Google Gemini API key
genai.configure(api_key="AIzaSyAKeZjJVpbkj5ic972h41JPxV33SDPa1-g")

# ✅ Use the correct Gemini model (text-to-image)
model = genai.GenerativeModel("models/imagegeneration")

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()

    # Safety check
    if not data or "prompt" not in data:
        app.logger.error("Missing prompt in request")
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    prompt = data["prompt"]
    app.logger.info(f"Received prompt: {prompt}")

    try:
        # Generate content using Gemini
        response = model.generate_content(prompt)
        app.logger.info("Gemini response received")

        # Extract base64 image from response
        image_data = response.candidates[0].content.parts[0].inline_data.data

        app.logger.info("Image data extracted successfully")

        return jsonify({"image_base64": image_data})
    except Exception as e:
        app.logger.error(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
