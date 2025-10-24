import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np

st.title("🐭 Tom & Jerry Smart Detector (Auto Sensitive Mode)")

# ==========================
# Load YOLO model
# ==========================
@st.cache_resource
def load_yolo():
    model = YOLO("model/Bulqis_Laporan_4.pt")
    return model

model = load_yolo()

# Ambil label asli dari model
LABELS = model.names  # <- label pasti sesuai model kamu
st.sidebar.write("Model Labels:", list(LABELS.values()))

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Asli", width=350)

    # Kolom sejajar: kiri gambar asli, kanan hasil deteksi
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Gambar Diupload", width="stretch")

    # ==========================
    # Auto Sensitive Detection
    # ==========================
    img_array = np.array(img)
    res1 = model.predict(img_array, conf=0.05, iou=0.5, imgsz=640, augment=True, verbose=False)

    # Flip horizontal detection (untuk pose miring/terbalik)
    flipped_img = ImageOps.mirror(img)
    flipped_array = np.array(flipped_img)
    res2 = model.predict(flipped_array, conf=0.05, iou=0.5, imgsz=640, augment=True, verbose=False)

    # Gabungkan hasil dua arah (ambil yang confidence tertinggi)
    result_img = np.maximum(res1[0].plot(), res2[0].plot())

    with col2:
        st.image(result_img, caption="Hasil Deteksi", width="stretch")

    # ==========================
    # Menampilkan Label Sesuai Model
    # ==========================
    all_boxes = list(res1[0].boxes) + list(res2[0].boxes)
    if all_boxes:
        st.markdown("### 🎯 Karakter Terdeteksi:")
        for box in all_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_name = LABELS[cls_id]  # <- ambil label langsung dari model
            st.write(f"- {label_name} ({conf:.2f})")
    else:
        st.warning("😕 Tidak ada karakter terdeteksi.")
