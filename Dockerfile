FROM python:3.10-slim

# Install dependensi sistem untuk OpenCV dan pemrosesan citra
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Buat user khusus non-root untuk keamanan Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Salin & install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Salin seluruh kode aplikasi
COPY --chown=user . .

# Buat folder sementara untuk upload gambar
RUN mkdir -p tmp_uploads

# Port default Hugging Face Spaces
EXPOSE 7860

# Jalankan Flask API
CMD ["python", "api_server.py"]
