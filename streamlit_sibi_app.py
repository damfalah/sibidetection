import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import numpy as np
import cv2
import joblib
import mediapipe as mp
from collections import deque, Counter

# === Konfigurasi halaman utama ===
st.set_page_config(page_title="Deteksi SIBI", layout="wide")

# === Sidebar Interaktif ===
with st.sidebar:
    st.image("assets/img/alfabet.jpg", width=180)
    st.markdown("## Menu Utama")
    halaman = st.radio("📂 Pilih Halaman:", ["🏠 Beranda", "📷 Deteksi SIBI"])

# === Halaman: Beranda ===
if halaman.startswith("🏠"):
    st.title("Abjad Bahasa Isyarat SIBI")
    st.markdown("""
    Aplikasi ini mendeteksi huruf-huruf **SIBI (Sistem Isyarat Bahasa Indonesia)** secara real-time.  
    Menggunakan MediaPipe dan model klasifikasi Random Forest yang **dilatih tanpa normalisasi**.

    > Gunakan **tangan kanan**, posisikan di depan kamera untuk mengenali huruf A–Y.  
    Huruf **J** dan **Z** tidak didukung karena melibatkan gerakan dinamis.
    """)
    st.image("assets/img/alfabet.jpg", caption="Abjad dalam SIBI", use_container_width=True)
    st.info("Pindah ke halaman **Deteksi SIBI** melalui sidebar untuk memulai pengenalan huruf.")

# === Halaman: Deteksi SIBI ===
elif halaman.startswith("📷"):
    st.title("Deteksi Huruf SIBI Real-Time")

    # Load model
    model = joblib.load("sibi_rf_model.pkl")

    # Inisialisasi MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # === Video Processor ===
    class SignPredictor(VideoTransformerBase):
        def __init__(self):
            self.prediction_history = deque(maxlen=9)
            self.current_prediction = "..."
            self.last_appended = None
            self.kata = ""

        def extract_raw_landmarks(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                if coords.shape == (21, 3):
                    return coords.flatten()
            return None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            features = self.extract_raw_landmarks(img)
            if features is not None:
                try:
                    pred = model.predict([features])[0]
                    self.prediction_history.append(pred)

                    if len(self.prediction_history) >= 6:
                        most_common, count = Counter(self.prediction_history).most_common(1)[0]
                        if count >= 6:
                            self.current_prediction = most_common
                            if self.current_prediction != self.last_appended:
                                self.kata += self.current_prediction
                                self.last_appended = self.current_prediction
                        else:
                            self.current_prediction = "Menstabilkan..."
                    else:
                        self.current_prediction = "Menstabilkan..."
                except Exception as e:
                    self.current_prediction = f"Error: {str(e)}"
            else:
                self.prediction_history.clear()
                self.current_prediction = "Tangan tidak terdeteksi"

            # === Tampilkan huruf prediksi (kiri atas) ===
            cv2.rectangle(img, (0, 0), (400, 40), (0, 0, 0), -1)
            cv2.putText(img, f"{self.current_prediction}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # === Tampilkan kata (kiri bawah) ===
            kata_text = f"KATA = {self.kata if self.kata else ''}"
            (text_w, _), _ = cv2.getTextSize(kata_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            y1 = img.shape[0] - 50
            y2 = img.shape[0]
            cv2.rectangle(img, (0, y1), (text_w + 20, y2), (0, 0, 0), -1)
            cv2.putText(img, kata_text, (10, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            return img

    # Tampilkan kamera
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_ctx = webrtc_streamer(
            key="sibi-app",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignPredictor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    st.markdown("---")
    st.info("💡 Tips: Letakkan tangan kanan di tengah kamera dan tahan selama 1–2 detik. Untuk menghapus kata yang sudah terbentuk, silakan hentikan dan jalankan kembali kamera.")
    st.caption("Model: Random Forest | Skripsi SIBI | by Mohammad Adam Falah")
