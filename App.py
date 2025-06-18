import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import datetime
import pickle
from typing import cast, BinaryIO

# ==================== Konstanta ====================
MODEL_PATH = 'model/gender_model.pkl'
RIWAYAT_PATH = 'history/riwayat.csv'
USER_DATA_PATH = 'users.csv'

if not os.path.exists(USER_DATA_PATH):
    # Jika belum ada file, buat file baru dengan header kolom
    df = pd.DataFrame(columns=["username", "password", "email"])
    df.to_csv(USER_DATA_PATH, index=False)
    print("File users.csv berhasil dibuat.")

# ==================== Setup Direktori & Model Dummy ====================
os.makedirs('model', exist_ok=True)
os.makedirs('history', exist_ok=True)

# Buat model dummy jika belum ada
if not os.path.exists(MODEL_PATH):
    X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_dummy = ['laki-laki', 'perempuan', 'laki-laki', 'perempuan']
    from sklearn.svm import SVC
    dummy_model = SVC(probability=True)
    dummy_model.fit(X_dummy, y_dummy)
    with open(MODEL_PATH, 'wb') as f:
        f = cast(BinaryIO, f)
        pickle.dump(dummy_model, f)

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# ==================== Fungsi Login / Register ====================
def login(username, password):
    username = username.strip().lower()
    password = password.strip()

    try:
        df_users = pd.read_csv("users.csv")

        # Normalisasi kolom agar aman
        df_users['username'] = df_users['username'].astype(str).str.strip().str.lower()
        df_users['password'] = df_users['password'].astype(str).str.strip()

        user_row = df_users[
            (df_users['username'] == username) & (df_users['password'] == password)
        ]

        return not user_row.empty
    except Exception as e:
        print("Login Error:", e)
        return False


def register_user(username, password, email):
    import csv

    username = username.strip().lower()
    password = password.strip()
    email = email.strip().lower()

    file_path = "users.csv"

    # Buat file jika belum ada
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password", "email"])

    # Cek apakah username sudah ada
    with open(file_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"].strip().lower() == username:
                return False  # Username sudah dipakai

    # Tambahkan user baru
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password, email])

    return True

# ==================== Session State Init ====================
if 'login_status' not in st.session_state:
    st.session_state.login_status = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# ==================== Autentikasi ====================
if not st.session_state.login_status:
    st.set_page_config(page_title="Login", layout="centered")

    # CSS untuk mempercantik tampilan
    st.markdown("""
    <style>
        body {
            background-color: #FFF0F5;
        }
        .auth-box {
            background-color: #FFE4EC;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            max-width: 500px;
            margin: auto;
        }
        .auth-title {
            font-size: 24px;
            font-weight: bold;
            color: #C71585;
            text-align: center;
            margin-bottom: 10px;
        }
        .auth-subtitle {
            font-size: 14px;
            color: #8B008B;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #FF69B4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #FF1493;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Menu Navigasi
    menu = st.sidebar.radio("🔧 Menu", ["🔐 Login", "📝 Registrasi"])

    # Judul Halaman
    st.markdown("<h2 style='text-align:center; color:#C71585;'>🧠 Sistem Deteksi Jenis Kelamin</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8B008B;'>Silakan login atau registrasi untuk melanjutkan</p>", unsafe_allow_html=True)

    # === Konten Utama ===
    with st.container():
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)

        if menu == "🔐 Login":
            st.markdown('<div class="auth-title">🔐 Login</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">Masukkan akun Anda</div>', unsafe_allow_html=True)

            username = st.text_input("👤 Username", key="login_user")
            password = st.text_input("🔒 Password", type="password", key="login_pass")

            if st.button("🔓 Login"):
                if login(username, password):
                    st.session_state.login_status = True
                    st.session_state.username = username
                    st.success(f"Selamat datang, {username}!")
                    st.rerun()

                else:
                    st.error("❌ Username atau password salah.")

        elif menu == "📝 Registrasi":
            st.markdown('<div class="auth-title">📝 Registrasi</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">Buat akun baru</div>', unsafe_allow_html=True)

            new_username = st.text_input("👤 Username", key="reg_user")
            new_email = st.text_input("📧 Email", key="reg_email")
            new_password = st.text_input("🔐 Password", type="password", key="reg_pass")

            if st.button("✅ Registrasi"):
                if register_user(new_username, new_password, new_email):
                    st.success("🎉 Registrasi berhasil. Silakan login.")
                else:
                    st.warning("⚠️ Username sudah digunakan.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ==================== Fungsi Deteksi & Prediksi ====================
def detect_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return gray[y:y + h, x:x + w]
    return None

def predict_gender(face_img):
    face_img = cv2.resize(face_img, (100, 100)).flatten().reshape(1, -1)
    prediction = model.predict(face_img)
    proba = model.predict_proba(face_img).max()
    return prediction[0], round(proba * 100, 2)

# ==================== Halaman Utama ====================
st.set_page_config(page_title="Deteksi Jenis Kelamin", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #FFF0F5;
            color: #800040;
        }
        .main {
            background-color: #FFE4EC;
        }
        h1, h2, h3, .stMarkdown {
            color: #C71585;
        }
        .stButton>button {
            background-color: #FF69B4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #FF1493;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🎀 Navigasi")
st.sidebar.markdown(f"👤 Login sebagai: **{st.session_state.username}**")
if st.sidebar.button("🚪 Logout"):
    st.session_state.login_status = False
    st.session_state.username = ""
    st.success("Anda telah logout.")
    st.rerun()

menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Upload Gambar", "Riwayat"])

if menu == "Beranda":
    st.markdown("<h1 style='text-align: center;'>🧠 Sistem Deteksi Jenis Kelamin Otomatis</h1>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.markdown("### 🚀 Cara Menggunakan:")
    st.markdown("""
    1. Pilih menu **Upload Gambar** di samping kiri.
    2. Unggah atau ambil foto wajah yang jelas.
    3. Klik tombol **Proses Deteksi** untuk melihat hasil.
    4. Lihat riwayat hasil deteksi di menu **Riwayat**.
    """)
    st.markdown("---")
    st.markdown('<div style="text-align:center;font-style:italic;">Selamat mencoba! 💖</div>', unsafe_allow_html=True)

elif menu == "Upload Gambar":
    st.title("📸 Upload & Deteksi Wajah")

    input_method = st.radio("Pilih metode input gambar:", ("Upload File", "Kamera"))

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Unggah gambar wajah", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Gambar yang Diunggah", use_container_width=True)

            face = detect_face(img)
            if face is not None:
                st.success("Wajah berhasil terdeteksi.")
                if st.button("Proses Deteksi"):
                    gender, confidence = predict_gender(face)
                    st.write(f"**Jenis Kelamin:** {gender}")
                    st.write(f"**Keakuratan:** {confidence}%")
                    new_row = {
                        "Tanggal": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Jenis Kelamin": gender,
                        "Akurasi (%)": confidence,
                        "Pengguna": st.session_state.username
                    }
                    df = pd.DataFrame([new_row])
                    if os.path.exists(RIWAYAT_PATH):
                        df.to_csv(RIWAYAT_PATH, mode='a', header=False, index=False)
                    else:
                        df.to_csv(RIWAYAT_PATH, index=False)
            else:
                st.warning("❗ Wajah tidak terdeteksi.")

    elif input_method == "Kamera":
        photo = st.camera_input("Ambil foto menggunakan kamera")
        if photo:
            file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Foto dari kamera", use_container_width=True)

            face = detect_face(img)
            if face is not None:
                st.success("Wajah berhasil terdeteksi.")
                if st.button("Proses Deteksi Kamera"):
                    gender, confidence = predict_gender(face)

                    st.write(f"**Jenis Kelamin:** {gender}")
                    st.write(f"**Keakuratan:** {confidence}%")
                    new_row = {
                        "Tanggal": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Jenis Kelamin": gender,
                        "Akurasi (%)": confidence,
                        "Pengguna": st.session_state.username
                    }
                    df = pd.DataFrame([new_row])
                    if os.path.exists(RIWAYAT_PATH):
                        df.to_csv(RIWAYAT_PATH, mode='a', header=False, index=False)
                    else:
                        df.to_csv(RIWAYAT_PATH, index=False)
            else:
                st.warning("❗ Wajah tidak terdeteksi.")

elif menu == "Riwayat":
    st.markdown("## 📂 Riwayat Deteksi")
    st.markdown("---")

    if os.path.exists(RIWAYAT_PATH):
        df = pd.read_csv(RIWAYAT_PATH)
        user_df = df[df['Pengguna'] == st.session_state.username]

        if user_df.empty:
            st.info("🔎 Belum ada riwayat deteksi untuk akun ini.")
        else:
            st.markdown("### 📋 Riwayat Anda:")
            st.dataframe(user_df, use_container_width=True)

            with st.expander("🗑️ Hapus Riwayat Saya"):
                st.warning("Tindakan ini akan menghapus seluruh riwayat Anda secara permanen.", icon="⚠️")
                if st.button("Konfirmasi Hapus Riwayat", type="primary"):
                    df = df[df['Pengguna'] != st.session_state.username]
                    df.to_csv(RIWAYAT_PATH, index=False)
                    st.success("✅ Riwayat Anda telah dihapus.")
                    st.rerun()
    else:
        st.info("📂 Belum ada data riwayat deteksi yang tersedia.")
