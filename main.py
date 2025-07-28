from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Google Gemini API Key
genai.configure(api_key="AIzaSyAKeZjJVpbkj5ic972h41JPxV33SDPa1-g")

model = genai.GenerativeModel("models/gemini-pro-vision")

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")

    try:
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
