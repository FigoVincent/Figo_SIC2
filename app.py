import streamlit as st
import joblib
import os

@st.cache_resource
def load_local_model():
    # Cek file dulu (anti error)
    if not os.path.exists("kmeans_rfm_model.pkl"):
        raise FileNotFoundError("kmeans_rfm_model.pkl tidak ditemukan")

    if not os.path.exists("scaler_rfm.pkl"):
        raise FileNotFoundError("scaler_rfm.pkl tidak ditemukan")

    # Load model & scaler
    kmeans_model = joblib.load("kmeans_rfm_model.pkl")
    scaler = joblib.load("scaler_rfm.pkl")

    return kmeans_model, scaler


# =========================
# LOAD MODEL
# =========================
try:
    model, scaler = load_local_model()
    st.sidebar.success("✅ Model K-Means & Scaler berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    model, scaler = None, None



st.title("📊 RFM Customer Segmentation")

if model is not None:
    st.write("Model siap digunakan")

    # Contoh input
    recency = st.number_input("Recency (hari)", value=30)
    frequency = st.number_input("Frequency", value=5)
    monetary = st.number_input("Monetary", value=500.0)

    if st.button("Prediksi Cluster"):
        import numpy as np

        X = np.array([[recency, frequency, monetary]])
        X_scaled = scaler.transform(X)
        cluster = model.predict(X_scaled)

        st.success(f"Customer termasuk ke **Cluster {cluster[0]}**")
