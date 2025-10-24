import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Deteksi Karakter Tom & Jerry",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM NEON GRADIENT CSS -------------------
st.markdown("""
<style>
/* Gradient Neon untuk body, sidebar, container utama */
body, .main, .block-container, .css-1d391kg {
    background: linear-gradient(135deg, #00f0ff, #fff700) !important;
    color: #fff !important;
    font-weight: bold;
}

/* Sidebar text & background */
.css-1d391kg {
    background: linear-gradient(180deg, #00f0ff, #fff700) !important;
}

/* Tombol radio sidebar jadi lebih neon */
div[data-baseweb="radio"] label {
    background: rgba(255, 255, 255, 0.1);
    color: #fff;
    padding: 10px 15px;
    border-radius: 12px;
    margin-bottom: 5px;
    display: block;
    text-align: center;
    font-weight: bold;
    border: 2px solid #00f0ff;
    transition: all 0.3s ease;
}
div[data-baseweb="radio"] label:hover {
    background: rgba(255, 255, 255, 0.3);
    color: #001f3f;
    border-color: #fff700;
}

/* Uploader container */
div.stFileUpload>div>div {
    background: rgba(255,255,255,0.2) !important;
    color: #fff !important;
    border-radius: 12px;
    padding: 10px;
    font-weight: bold;
    border: 1px solid #00f0ff;
}

/* Tombol "Browse files" */
div.stFileUpload button {
    background: linear-gradient(45deg, #00f0ff, #fff700) !important;
    color: #001f3f !important;
    border-radius: 12px;
    font-weight: bold;
    border: 2px solid #fff;
    transition: all 0.3s ease;
}
div.stFileUpload button:hover {
    background: linear-gradient(45deg, #fff700, #00f0ff) !important;
    color: #001f3f !important;
}

/* Placeholder teks */
div.stFileUpload div[data-testid="stFileUploadDropzone"] {
    color: #fff !important;
}

/* Neon heading */
h1 {
    text-shadow: 0 0 5px #00f0ff, 0 0 10px #fff700, 0 0 20px #00f0ff;
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align:center;'>🎬 Deteksi Karakter Tom & Jerry</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Sistem deteksi otomatis karakter berbasis YOLOv8.</p>", unsafe_allow_html=True)

# ------------------- SIDEBAR NAVIGATION -------------------
menu = st.sidebar.radio("", ["ℹ️ Tentang", "🧠 Deteksi"])

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model_path = "model/Bulqis_Laporan_4.pt"
    model = YOLO(model_path)
    return model

# ------------------- TENTANG -------------------
if menu == "ℹ️ Tentang":
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

# ------------------- DETEKSI -------------------
elif menu == "🧠 Deteksi":
    st.subheader("🚀 Unggah Gambar untuk Deteksi & Klasifikasi Karakter")

    uploaded_file = st.file_uploader("Pilih gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((640, 480))  # resize agar lebih ringan

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)

        with col2:
            with st.spinner("Model sedang memproses gambar..."):
                try:
                    model = load_model()
                    results = model.predict(image, conf=0.5, imgsz=640, verbose=False)
                    result_image = results[0].plot()  # hasil deteksi ke array numpy

                    st.image(result_image, caption="Hasil Deteksi", use_container_width=True)

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
