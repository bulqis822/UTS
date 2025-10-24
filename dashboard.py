import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Deteksi Karakter Tom & Jerry",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Uploader container */
div.stFileUpload>div>div {
    background-color: white !important;  /* background putih */
    color: #001f3f !important;           /* teks gelap */
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
}

/* Tombol "Browse files" */
div.stFileUpload button {
    background-color: #ff851b !important; /* oranye */
    color: white !important;              /* teks putih */
    border-radius: 8px;
    font-weight: bold;
}

div.stFileUpload button:hover {
    background-color: #ffaa33 !important; /* oranye muda saat hover */
    color: #001f3f !important;
}

/* Teks placeholder "Drag and drop file here" */
div.stFileUpload div[data-testid="stFileUploadDropzone"] {
    color: #001f3f !important;  /* teks gelap */
}
</style>
""", unsafe_allow_html=True)



# ------------------- HEADER -------------------
st.markdown("<h1>🎬 Deteksi Karakter Tom & Jerry</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Sistem deteksi otomatis karakter berdasarkan model YOLO yang kamu latih sendiri.</p>", unsafe_allow_html=True)

# ------------------- SIDEBAR NAVIGATION -------------------
menu = st.sidebar.radio("Navigasi", ["🧠 Deteksi", "ℹ️ Tentang"])

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model_path = "model/Bulqis_Laporan_4.pt"
    model = YOLO(model_path)
    return model

# ------------------- DETEKSI -------------------
if menu == "🧠 Deteksi":
    st.subheader("🚀 Unggah Gambar untuk Deteksi & Klasifikasi Karakter")

    uploaded_file = st.file_uploader("Pilih gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((640, 480))  # resize agar lebih ringan

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.image(image, caption="Gambar Asli", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            with st.spinner("Model sedang memproses gambar..."):
                try:
                    model = load_model()
                    results = model.predict(image, conf=0.5, imgsz=640, verbose=False)
                    result_image = results[0].plot()  # hasil deteksi ke array numpy

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # tampilkan label hasil deteksi
                    detected_labels = set()
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        label = model.names[cls]
                        detected_labels.add(label)

                    if detected_labels:
                        st.success(f"Karakter terdeteksi: {', '.join(detected_labels)}")
                    else:
                        st.warning("Tidak ada karakter yang terdeteksi.")

                except Exception as e:
                    st.error(f"Gagal menjalankan deteksi: {e}")
    else:
        st.info("📂 Silakan unggah gambar terlebih dahulu untuk mendeteksi karakter.")

# ------------------- TENTANG -------------------
elif menu == "ℹ️ Tentang":
    st.subheader("ℹ️ Tentang Aplikasi Ini")
    st.markdown("""
    Aplikasi ini dibuat untuk **mendeteksi karakter Tom dan Jerry** secara otomatis menggunakan model **YOLOv8**.
    
    ### 🧩 Cara Menggunakan:
    1. Masuk ke menu **Deteksi** di sidebar.
    2. Unggah gambar yang berisi karakter Tom atau Jerry.
    3. Tunggu sebentar hingga model selesai memproses.
    4. Hasil deteksi akan muncul di samping, lengkap dengan nama karakter.
    """)
    st.markdown("---")
    st.markdown("""
    ### 👩‍💻 Pembuat:
    Dibuat oleh **Bulqis** — mahasiswa Statistika, Universitas Syiah Kuala.  
    """)
