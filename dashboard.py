import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Bulqis Object Detection", page_icon="🤖", layout="wide")

st.title("🎬 Tom and Jerry Object Detection")
st.markdown("Upload gambar atau video untuk mendeteksi karakter menggunakan model YOLO (.pt).")

# === Load YOLO model ===
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join("model", "Bulqis_Laporan_4.pt")
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

model = load_model()

# === Pilihan Mode ===
mode = st.radio("Pilih Mode Deteksi:", ["Gambar", "Video"], horizontal=True)

if model:
    if mode == "Gambar":
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            results = model.predict(img_array, conf=0.4, iou=0.5)
            
            annotated_frame = results[0].plot()
            st.image(annotated_frame, caption="Hasil Deteksi", use_container_width=True)

            # Tampilkan label dan confidence
            st.subheader("Deteksi Terdeteksi:")
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                st.write(f"- **{label}** ({conf:.2f})")

    elif mode == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            st.video(tfile.name)
            st.info("⏳ Tunggu sebentar, sedang memproses video...")

            cap = cv2.VideoCapture(tfile.name)
            output_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, conf=0.4, iou=0.5)
                annotated_frame = results[0].plot()
                output_frames.append(annotated_frame)
            
            cap.release()
            st.success("✅ Deteksi selesai!")
            st.image(output_frames, caption="Hasil Frame Deteksi", use_container_width=True)

else:
    st.warning("Model belum dimuat, periksa nama file dan path model di folder 'model/'.")
