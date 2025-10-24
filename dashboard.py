import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==========================
# CONFIG STREAMLIT
# ==========================
st.set_page_config(page_title="Deteksi Karakter Tom & Jerry 🐭🐱", layout="wide")

st.title("🎬 Deteksi Karakter Tom & Jerry")
st.write("Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi karakter **Tom** dan **Jerry** pada gambar secara otomatis.")

# ==========================
# LOAD MODEL YOLO
# ==========================
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("model/Bulqis_Laporan_4.pt")  # pastikan path sesuai
        st.sidebar.success("✅ Model YOLO berhasil dimuat!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat model YOLO: {e}")
        return None

model = load_yolo_model()

if model is None:
    st.stop()

# Ambil label dari model (harus sesuai training)
LABELS = model.names

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("Unggah gambar yang ingin dideteksi:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file).convert("RGB")

    # Konversi ke numpy array untuk pemrosesan YOLO
    image_np = np.array(image_uploaded)

    # Jalankan deteksi YOLO
    with st.spinner("🔍 Sedang mendeteksi objek..."):
        results = model.predict(image_np, conf=0.45, iou=0.5, verbose=False)  
        # conf=0.45 agar lebih peka, tapi tetap akurat

    result_img = results[0].plot()  # hasil deteksi (dalam format OpenCV BGR)

    # Konversi ke RGB agar bisa ditampilkan Streamlit
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # ==========================
    # TAMPILKAN HASIL (SAMPING-SAMPING)
    # ==========================
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_uploaded, caption="📤 Gambar yang Diupload", width=400)

    with col2:
        st.image(result_img_rgb, caption="✅ Hasil Deteksi YOLO", width=400)

    # ==========================
    # TAMPILKAN INFO DETEKSI
    # ==========================
    st.subheader("📊 Detil Deteksi:")
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = LABELS.get(cls_id, "Unknown")
            conf = float(box.conf[0])
            st.write(f"- **{label}** (Confidence: {conf:.2f})")
    else:
        st.warning("⚠️ Tidak ada karakter yang terdeteksi di gambar ini.")
else:
    st.info("📁 Silakan unggah gambar terlebih dahulu untuk mendeteksi karakter.")
