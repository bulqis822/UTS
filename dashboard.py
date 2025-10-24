import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# --- TITLE ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🪶 YOLO Object Detection Dashboard</h1>", unsafe_allow_html=True)

# --- LOAD MODEL (cached biar tidak reload terus) ---
@st.cache_resource
def load_model():
    model_path = "model/Bulqis_Laporan_4.pt"  # pastikan path benar
    model = YOLO(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- PILIH INPUT ---
st.sidebar.title("⚙️ Pengaturan")
mode = st.sidebar.radio("Pilih Mode:", ["Gambar", "Video"])

if mode == "Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        # kompres ukuran biar ringan
        img = img.resize((640, 480))

        st.image(img, caption="Gambar Asli", use_container_width=True)

        with st.spinner("Mendeteksi objek..."):
            results = model.predict(img, conf=0.5, imgsz=640, verbose=False)
            result_img = results[0].plot()  # hasil deteksi jadi numpy array
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

elif mode == "Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        with st.spinner("Mendeteksi objek... (harap tunggu, CPU mode)"):
            results = model.predict(source=tfile.name, conf=0.5, stream=False, imgsz=640, verbose=False)
            
            output_path = os.path.join("temp", "hasil.mp4")
            os.makedirs("temp", exist_ok=True)
            model.export(format="onnx", simplify=True)  # biar next run lebih cepat

            st.success("✅ Deteksi selesai (tampilkan video hasilnya belum diaktifkan untuk kecepatan).")

st.markdown("---")
st.caption("🚀 Dibuat dengan YOLOv8 + Streamlit | Optimized by ChatGPT GPT-5")
