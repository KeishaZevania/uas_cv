
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# pyrefly: ignore [missing-import]
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# pyrefly: ignore [missing-import]
import kagglehub

IMG_SIZE = (128, 128)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. DOWNLOAD DATASET DARI KAGGLE
# ---------------------------------------------------------------------------
def download_dataset():
    path = kagglehub.dataset_download(
        "hasbimubarak/soil-moisture-dataset-based-on-measurement"
    )
    print("Path to dataset files:", path)
    return path


# ---------------------------------------------------------------------------
# 2. CARI SEMUA FILE GAMBAR + LABEL DARI NAMA FOLDER
# ---------------------------------------------------------------------------
def collect_image_paths(dataset_path):
    """
    Asumsi struktur folder dataset:
        dataset_path/
            <kategori_1>/*.jpg
            <kategori_2>/*.jpg
            ...
    Kategori bisa berupa nama kelas asli dataset (misal berdasarkan nilai
    kelembaban). Skrip ini otomatis mendeteksi semua sub-folder sebagai label.
    Jika dataset kamu menggunakan struktur berbeda (misal CSV index),
    sesuaikan fungsi ini.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    data = []
    for root, dirs, files in os.walk(dataset_path):
        for ext in exts:
            for filepath in glob(os.path.join(root, ext)):
                label = os.path.basename(os.path.dirname(filepath))
                data.append({"filepath": filepath, "label": label})
    df = pd.DataFrame(data)
    print("Jumlah gambar ditemukan:", len(df))
    print("Distribusi label asli dataset:\n", df["label"].value_counts())
    return df


def map_to_3_classes(label_raw, mapping=None):
    """
    Dataset Kaggle ini berisi label berdasarkan nilai pengukuran kelembaban
    (numerik/persentase), sedangkan proyek kita butuh 3 kelas:
    kering / lembab / basah.
    """
    label_str = str(label_raw).strip()
    if label_str == "0-10" or label_str == "10-20":
        return "kering"
    elif label_str == "20-40":
        return "lembab"
    elif label_str == "40-100":
        return "basah"

    try:
        val = float(label_str.replace("%", "").strip())
        if val < 30:
            return "kering"
        elif val < 60:
            return "lembab"
        else:
            return "basah"
    except ValueError:
        # fallback: cocokkan berdasarkan kata kunci nama folder
        label_lower = label_str.lower()
        if "dry" in label_lower or "kering" in label_lower:
            return "kering"
        elif "wet" in label_lower or "basah" in label_lower:
            return "basah"
        else:
            return "lembab"



# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # noise reduction
    return img


# ---------------------------------------------------------------------------
# 4. SEGMENTASI SEDERHANA (opsional, fokus ke area tanah)
# ---------------------------------------------------------------------------
def simple_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented


# ---------------------------------------------------------------------------
# 5. EKSTRAKSI FITUR: WARNA (HSV) + TEKSTUR (GABOR FILTER)
# ---------------------------------------------------------------------------
def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        feats.append(hsv[:, :, ch].mean())
        feats.append(hsv[:, :, ch].std())
    return feats  # 6 fitur: H_mean,H_std,S_mean,S_std,V_mean,V_std


def build_gabor_kernels():
    kernels = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        for frequency in (0.1, 0.25, 0.4):
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            kernels.append(kernel)
    return kernels


GABOR_KERNELS = build_gabor_kernels()


def extract_gabor_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    feats = []
    for kernel in GABOR_KERNELS:
        filtered = ndi.convolve(gray, kernel, mode="wrap")
        feats.append(filtered.mean())
        feats.append(filtered.var())
    return feats  # 4 orientasi x 3 frekuensi x 2 (mean,var) = 24 fitur


def extract_features(filepath):
    img = preprocess_image(filepath)
    if img is None:
        return None
    img = simple_segmentation(img)
    color_feats = extract_color_features(img)
    gabor_feats = extract_gabor_features(img)
    return color_feats + gabor_feats


# ---------------------------------------------------------------------------
# 6. BUILD DATASET FITUR
# ---------------------------------------------------------------------------
def build_feature_dataset(df):
    X, y, valid_idx = [], [], []
    for i, row in df.iterrows():
        feats = extract_features(row["filepath"])
        if feats is not None:
            X.append(feats)
            y.append(row["label_3class"])
            valid_idx.append(i)
        if i % 50 == 0:
            print(f"Diproses {i}/{len(df)}")
    X = np.array(X)
    y = np.array(y)
    return X, y


COLOR_COLS = ["H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std"]
GABOR_COLS = [f"gabor_{i}" for i in range(24)]
FEATURE_COLUMNS = COLOR_COLS + GABOR_COLS


# ---------------------------------------------------------------------------
# 7. TRAIN + EVALUASI
# ---------------------------------------------------------------------------
def train_and_evaluate(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=le.classes_
    ))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix - Klasifikasi Kelembaban Tanah")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix disimpan ke confusion_matrix.png")

    return clf, le, scaler


# ---------------------------------------------------------------------------
# 8. SIMPAN MODEL UNTUK DIPAKAI API
# ---------------------------------------------------------------------------
def save_artifacts(clf, le, scaler, out_dir="model_artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print(f"Model tersimpan di folder: {out_dir}/")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = download_dataset()

    df = collect_image_paths(dataset_path)

    # >>> PENTING: cek dulu hasil df['label'].value_counts() di atas,
    # lalu sesuaikan fungsi map_to_3_classes() agar pas dengan label asli
    # dataset (apakah berupa angka persentase, atau nama kategori lain).
    df["label_3class"] = df["label"].apply(map_to_3_classes)
    print("\nDistribusi setelah dipetakan ke 3 kelas:\n",
          df["label_3class"].value_counts())

    X, y = build_feature_dataset(df)

    feat_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    feat_df["label"] = y
    feat_df.to_csv("extracted_features.csv", index=False)
    print("Fitur hasil ekstraksi disimpan ke extracted_features.csv")

    clf, le, scaler = train_and_evaluate(X, y)
    save_artifacts(clf, le, scaler)
