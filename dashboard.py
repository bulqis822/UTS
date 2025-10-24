import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Deteksi Karakter Tom & Jerry", layout="wide")

st.title("🎬 Deteksi Karakter Tom dan Jerry")
st.write("Unggah gambar untuk mendeteksi karakter berdasarkan model YOLO kamu.")

# Load model (pastikan path sesuai)
MODEL_PATH = "model/Bulqis_Laporan_4.pt"

@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Gambar Asli")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        if model is not None:
            st.subheader("🔍 Hasil Deteksi")

            # Simpan file sementara
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Deteksi otomatis
            results = model.predict(tmp_path, conf=0.35, iou=0.5)  # agak peka tapi tetap akurat
            result_image = results[0].plot()  # render deteksi ke array numpy

            # Konversi untuk Streamlit
            st.image(result_image, caption="Hasil Deteksi", use_column_width=True)

            # Tampilkan label deteksi
            detected_labels = set()
            for box in results[0].boxes:
                cls = int(box.cls)
                label = model.names[cls]
                detected_labels.add(label)
            
            if detected_labels:
                st.success(f"Karakter terdeteksi: {', '.join(detected_labels)}")
            else:
                st.warning("Tidak ada karakter yang terdeteksi di gambar ini.")
            
            os.remove(tmp_path)
        else:
            st.error("Model tidak berhasil dimuat. Periksa kembali file .pt kamu.")
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai deteksi.")
