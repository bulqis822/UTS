import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_yolo():
    try:
        model = YOLO("model/Bulqis_Laporan_4.pt")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

@st.cache_resource
def load_tf():
    try:
        model = tf.keras.models.load_model("model/Putri_Laporan_2.h5")
        return model
    except Exception as e:
        st.warning(f"Model TensorFlow tidak ditemukan: {e}")
        return None

yolo_model = load_yolo()
tf_model = load_tf()

# ==========================
# LABEL MODEL YOLO
# ==========================
if yolo_model is not None:
    LABELS = yolo_model.names  # label mengikuti model .pt
else:
    LABELS = {}

# ==========================
# UI STREAMLIT
# ==========================
st.set_page_config(page_title="Dashboard Deteksi Otomatis", layout="wide")

st.title("📷 Dashboard Deteksi Otomatis")
st.markdown("Upload gambar untuk mendeteksi objek secara otomatis dan akurat.")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    if yolo_model is not None:
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model.predict(img_np, conf=0.25)
            annotated_img = results[0].plot()  # gambar hasil deteksi dengan bounding box

        st.image(annotated_img, caption="Hasil Deteksi YOLO", use_column_width=True)

        # tampilkan label dan confidence
        det_table = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = LABELS.get(cls, "Unknown")
            det_table.append({"Label": label, "Confidence": round(conf, 2)})

        if det_table:
            st.subheader("📊 Hasil Deteksi:")
            st.dataframe(det_table)
        else:
            st.info("Tidak ada objek terdeteksi.")

        # jika ada model TF, gunakan untuk prediksi lanjutan
        if tf_model is not None:
            st.subheader("🧠 Prediksi Tambahan (Model TensorFlow)")
            img_resized = cv2.resize(img_np, (224, 224))
            img_resized = np.expand_dims(img_resized / 255.0, axis=0)
            preds = tf_model.predict(img_resized)
            predicted_class = np.argmax(preds, axis=1)[0]
            st.success(f"Prediksi tambahan: {predicted_class}")

    else:
        st.error("Model YOLO belum dimuat dengan benar.")
else:
    st.info("Silakan upload gambar terlebih dahulu.")

# ==========================
# ABOUT
# ==========================
st.sidebar.header("ℹ️ Tentang")
st.sidebar.write("""
Aplikasi ini menggunakan dua model:
- **YOLOv8** (`Bulqis_Laporan_4.pt`) untuk deteksi objek
- **TensorFlow** (`Putri_Laporan_2.h5`) untuk klasifikasi tambahan  
Dibuat otomatis agar hasil deteksi tidak acak dan sesuai label aslinya.
""")
