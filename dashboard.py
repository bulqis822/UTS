import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

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
st.title("🧠 Image Classification & Object Detection Dashboard")

st.sidebar.header("🔍 Pilih Mode")
menu = st.sidebar.radio("Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Buat layout dua kolom sejajar
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, caption="📸 Gambar Asli", width="stretch")

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            with st.spinner("Sedang mendeteksi objek..."):
                img_array = np.array(img)  # konversi ke array untuk hasil lebih akurat
                # Lebih peka dengan turunkan conf threshold
                results = yolo_model.predict(
                    source=img_array, conf=0.1, iou=0.5, verbose=False
                )

                result_img = results[0].plot()
                labels = results[0].names

                st.image(result_img, caption="✅ Hasil Deteksi", width="stretch")

                st.subheader("📋 Objek Terdeteksi:")
                detected = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    detected.append(f"- {labels[cls_id]} ({conf:.2f})")
                if detected:
                    st.write("\n".join(detected))
                else:
                    st.warning("Tidak ada objek yang terdeteksi 😕")

        elif menu == "Klasifikasi Gambar":
            with st.spinner("Sedang melakukan klasifikasi..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Ganti label ini sesuai model kamu
                labels = ["Kelas 1", "Kelas 2", "Kelas 3"]
                predicted_label = (
                    labels[class_index]
                    if class_index < len(labels)
                    else f"Class {class_index}"
                )

                st.image(img, caption="🧩 Klasifikasi", width="stretch")
                st.success(f"### Hasil Prediksi: {predicted_label}")
                st.write(f"Probabilitas: **{confidence:.2f}**")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memulai.")
