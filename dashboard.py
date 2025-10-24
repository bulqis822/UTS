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

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #001f3f;  /* biru dongker */
    color: #ff851b;             /* orange */
    font-family: 'Verdana', sans-serif;
}

/* Sidebar radio buttons */
.css-1avcm0n {
    color: #ff851b;
    font-weight: bold;
}

/* Main background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #87CEFA, #FFD580); /* biru muda ke orange muda */
    color: #001f3f;
}

/* Header */
h1 {
    color: #ff851b;
    font-family: 'Arial Black', sans-serif;
    text-align: center;
}

/* Subheaders */
h2, h3, h4 {
    color: #001f3f;
}

/* Buttons */
.stButton>button {
    background-color: #ff851b;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ffaa33;
    color: #001f3f;
}

/* Cards / box effect for result images */
.result-card {
    background-color: rgba(255, 255, 255, 0.7);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
}

/* File uploader */
div.stFileUpload > div > div > div {
    background-color: #FFD580 !important;  /* orange muda */
    color: #001f3f !important;            /* teks biru dongker */
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
}

/* Uploader button */
div.stFileUpload button {
    background-color: #ff851b !important; /* oranye terang */
    color: white !important;
    border-radius: 8px;
    font-weight: bold;
}

div.stFileUpload button:hover {
    background-color: #ffaa33 !important;
    color: #001f3f !important;
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
