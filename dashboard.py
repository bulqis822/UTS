import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Bulqis_Laporan_4.pt")  # Model YOLOv8
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/Putri_Laporan_2.h5")  # Model Klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==========================
# UI Layout
# ==========================
st.title("🧠 Image Classification & Object Detection Dashboard")

st.sidebar.header("🔍 Pilih Mode")
menu = st.sidebar.radio("Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)" and yolo_model:
        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi
            labels = results[0].names

            st.image(result_img, caption="✅ Hasil Deteksi", use_container_width=True)

            st.subheader("📋 Objek Terdeteksi:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                st.write(f"- {labels[cls_id]} ({float(box.conf[0]):.2f} confidence)")

    elif menu == "Klasifikasi Gambar" and classifier:
        with st.spinner("Sedang melakukan klasifikasi..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Misalnya kamu punya daftar label
            labels = ["Kelas 1", "Kelas 2", "Kelas 3"]  # Ganti sesuai model kamu

            predicted_label = labels[class_index] if class_index < len(labels) else f"Class {class_index}"

            st.success(f"### 🧩 Hasil Prediksi: {predicted_label}")
            st.write(f"Probabilitas: **{confidence:.2f}**")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memulai.")
