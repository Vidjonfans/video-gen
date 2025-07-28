from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# ✅ Google Gemini API key
genai.configure(api_key="AIzaSyAKeZjJVpbkj5ic972h41JPxV33SDPa1-g")  # अपनी असली API key यहाँ डालें

# ✅ Use the correct image generation model
model = genai.GenerativeModel("models/imagegeneration")

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")

    try:
        response = model.generate_content(prompt)
        
        # Extract the image data from response
        image_data = response.candidates[0].content.parts[0].inline_data.data
        
        return jsonify({"image_base64": image_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
