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
/* Body & aplikasi */
body, .stApp {
    background: radial-gradient(circle at top left, #0a1a0a, #0d2610, #112f12);
    color: #e0ffe0;
    font-family: 'Poppins', sans-serif;
    transition: all 0.3s ease;
}

/* Heading */
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}
h1 {
    font-size: 3rem;
    background: linear-gradient(90deg, #00ff4c, #ffff00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    animation: glowHeader 2s infinite alternate;
}
@keyframes glowHeader {
    0% { text-shadow: 0 0 5px #00ff4c, 0 0 10px #ffff00; }
    50% { text-shadow: 0 0 15px #00ff4c, 0 0 25px #ffff00; }
    100% { text-shadow: 0 0 5px #00ff4c, 0 0 10px #ffff00; }
}

/* Sidebar neon radio */
div[role="radiogroup"] > label {
    display: block;
    background: linear-gradient(90deg, #00ff4c, #ffff00);
    color: #001f00;
    font-weight: 700;
    padding: 12px 15px;
    margin-bottom: 12px;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 10px #00ff4c80, 0 0 20px #ffff4080;
}
div[role="radiogroup"] > label:hover {
    box-shadow: 0 0 25px #00ff4c, 0 0 50px #ffff00;
    transform: translateX(5px);
}
div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + span {
    box-shadow: 0 0 35px #00ff4c, 0 0 70px #ffff00;
    animation: neonPulse 1.5s infinite alternate;
}

/* Card hasil deteksi */
.result-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    margin-top: 20px;
    transition: all 0.4s ease-in-out;
    animation: neonCard 1.8s infinite alternate;
}
.result-card:hover {
    box-shadow: 0 0 25px #00ff4c80, 0 0 50px #ffff4060;
    transform: translateY(-5px);
}
@keyframes neonCard {
    0% { box-shadow: 0 0 10px #00ff4c60, 0 0 20px #ffff0060; }
    50% { box-shadow: 0 0 25px #00ff4c80, 0 0 50px #ffff4080; }
    100% { box-shadow: 0 0 10px #00ff4c60, 0 0 20px #ffff0060; }
}

/* Info box */
.info-box {
    background: linear-gradient(90deg, #0f1f0f, #1b2a12);
    border-radius: 12px;
    padding: 14px 20px;
    color: #bdfebd;
    text-align: center;
    margin-top: 10px;
    animation: neonPulse 1.5s infinite alternate;
}
@keyframes neonPulse {
    0% { box-shadow: 0 0 5px #00ff4c, 0 0 10px #ffff00; }
    50% { box-shadow: 0 0 15px #00ff4c, 0 0 25px #ffff00; }
    100% { box-shadow: 0 0 5px #00ff4c, 0 0 10px #ffff00; }
}

/* Subtext */
.subtext {
    font-size: 1rem;
    color: #bdfebd;
    margin-bottom: 20px;
}

/* Footer */
footer {
    background: linear-gradient(90deg, #0d120a, #1e2410);
    padding: 15px 20px;
    border-radius: 12px;
    margin-top: 50px;
    text-align: center;
    font-size: 0.9rem;
    color: #bdfebd;
    box-shadow: 0 0 20px #00ff4c40;
}

/* Responsive */
@media only screen and (max-width: 1024px) {
    .stColumns { flex-direction: column !important; }
    .stButton>button { margin-bottom: 15px; width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align:center;'>🎬 Deteksi Karakter Tom & Jerry</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Sistem deteksi otomatis karakter berbasis YOLOv8.</p>", unsafe_allow_html=True)

# ------------------- SIDEBAR NAVIGATION -------------------
menu = st.sidebar.radio(
    "Navigasi",
    ["ℹ️ Tentang", "🧠 Deteksi"],
    index=0,
    key="sidebar_menu"
)

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
        image = image.resize((640, 480))

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)

        with col2:
            with st.spinner("Model sedang memproses gambar..."):
                try:
                    model = load_model()
                    results = model.predict(image, conf=0.5, imgsz=640, verbose=False)
                    result_image = results[0].plot()

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
