import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Bulqis_Laporan_4.pt")
    classifier = tf.keras.models.load_model("model/Putri_Laporan_2.h5")
    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==========================
# UI Layout
# ==========================
st.title("🧠 Tom & Jerry Character Detection & Classification")

st.sidebar.header("🔍 Pilih Mode")
menu = st.sidebar.radio("Mode:", ["Deteksi Karakter (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, caption="📸 Gambar Asli", width="stretch")

    with col2:
        if menu == "Deteksi Karakter (YOLO)":
            with st.spinner("🔍 Mendeteksi karakter (otomatis lebih sensitif)..."):
                img_array = np.array(img)

                # ==== Inference utama ====
                results_main = yolo_model.predict(
                    source=img_array,
                    conf=0.1,  # lebih peka terhadap pose kecil/samping
                    iou=0.6,   # longgar supaya bounding box lebih toleran
                    imgsz=640, # multi-scale friendly
                    augment=True,
                    verbose=False
                )

                # ==== Inference tambahan (flip horizontal) ====
                flipped_img = ImageOps.mirror(img)
                flipped_array = np.array(flipped_img)
                results_flip = yolo_model.predict(
                    source=flipped_array,
                    conf=0.1,
                    iou=0.6,
                    imgsz=640,
                    augment=True,
                    verbose=False
                )

                # Gabungkan hasil (main + flip)
                result_img = results_main[0].plot()
                if len(results_flip[0].boxes) > 0:
                    flipped_plot = results_flip[0].plot()
                    result_img = np.maximum(result_img, flipped_plot)  # gabung hasil dua arah

                labels = results_main[0].names
                st.image(result_img, caption="✅ Hasil Deteksi (lebih peka)", width="stretch")

                # Tampilkan daftar objek
                all_boxes = list(results_main[0].boxes) + list(results_flip[0].boxes)
                if all_boxes:
                    st.subheader("📋 Karakter Terdeteksi:")
                    for box in all_boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"- {labels[cls_id]} ({conf:.2f})")
                else:
                    st.warning("Tidak ada karakter yang terdeteksi 😕")

        elif menu == "Klasifikasi Gambar":
            with st.spinner("🧠 Mengklasifikasikan gambar..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                labels = ["Tom", "Jerry", "Lainnya"]
                predicted_label = (
                    labels[class_index]
                    if class_index < len(labels)
                    else f"Class {class_index}"
                )

                st.image(img, caption="🧩 Hasil Klasifikasi", width="stretch")
                st.success(f"### Hasil Prediksi: {predicted_label}")
                st.write(f"Probabilitas: **{confidence:.2f}**")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memulai.")
