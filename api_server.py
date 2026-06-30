
import os
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from predict import predict_image

app = Flask(__name__, template_folder="templates")
CORS(app)  # supaya Laravel (domain berbeda) bisa akses API ini

UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Field 'image' tidak ditemukan"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        hasil = predict_image(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(hasil)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    # Hugging Face menggunakan port 7860 secara default
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)

