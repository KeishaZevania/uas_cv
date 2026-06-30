
import sys
import joblib
import numpy as np
from train_model import extract_features  # reuse fungsi ekstraksi fitur

MODEL_DIR = "model_artifacts"


def load_artifacts():
    clf = joblib.load(f"{MODEL_DIR}/model.pkl")
    le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return clf, le, scaler


def predict_image(filepath):
    clf, le, scaler = load_artifacts()
    feats = extract_features(filepath)
    if feats is None:
        return {"error": "Gambar tidak dapat dibaca"}

    feats_scaled = scaler.transform(np.array(feats).reshape(1, -1))
    pred_idx = clf.predict(feats_scaled)[0]
    pred_label = str(le.inverse_transform([pred_idx])[0])
    proba = clf.predict_proba(feats_scaled)[0]

    result = {
        "prediksi": pred_label,
        "probabilitas": {
            str(label): round(float(p), 4)
            for label, p in zip(le.classes_, proba)
        }
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python predict.py <path_gambar>")
        sys.exit(1)

    filepath = sys.argv[1]
    hasil = predict_image(filepath)
    print(hasil)
