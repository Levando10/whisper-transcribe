from flask import Flask, request, jsonify
import whisper
import requests
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("tiny")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    video_url = data.get("videoUrl")

    response = requests.get(video_url)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(response.content)
        f.flush()
        result = model.transcribe(f.name)

    return jsonify({"transcript": result["text"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
