

import streamlit as st
import pandas as pd
import numpy as np
import datetime

try:
    from src.inference import predict_booking_status, expected_columns # Impor juga expected_columns jika memungkinkan
except ImportError:
    st.error("Gagal mengimpor fungsi prediksi. Pastikan file src/inference.py ada dan dapat diakses.")
    st.stop()
    
st.set_page_config(
    page_title="Prediksi Pembatalan Booking Hotel",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Prediksi Status Pembatalan Booking Hotel")
st.markdown("""
Aplikasi ini memprediksi apakah sebuah pemesanan hotel kemungkinan akan dibatalkan atau tidak,
berdasarkan detail pemesanan yang dimasukkan.
Masukkan detail pemesanan di bawah ini dan klik **'Prediksi Status Booking'**.
""")

meal_plan_options = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'] 
room_type_options = ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
market_segment_options = ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation']

# --- Form Input ---
with st.form("booking_input_form"):
    st.subheader("Detail Tamu & Menginap")
    col1, col2, col3 = st.columns(3)
    with col1:
        no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, max_value=10, value=2, step=1)
        no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0, max_value=10, value=1, step=1)
    with col2:
        no_of_children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0, step=1)
        no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0, max_value=20, value=2, step=1) # Sesuaikan max value jika perlu
    with col3:
        # Menggunakan Radio untuk Yes/No agar lebih jelas
        required_car_parking_space_input = st.radio("Perlu Tempat Parkir?", ("Tidak", "Ya"), index=0)
        repeated_guest_input = st.radio("Tamu Berulang?", ("Tidak", "Ya"), index=0)

    st.divider()
    st.subheader("Detail Pemesanan")
    col4, col5, col6 = st.columns(3)
    with col4:
        type_of_meal_plan = st.selectbox("Jenis Paket Makan", options=meal_plan_options, index=0)
        # Menggunakan date input lebih user-friendly
        arrival_date_input = st.date_input("Tanggal Kedatangan", value=datetime.date.today() + datetime.timedelta(days=30)) # Default 30 hari dari sekarang
        lead_time = (arrival_date_input - datetime.date.today()).days # Hitung lead time otomatis
        st.write(f"Lead Time: **{lead_time}** hari") # Tampilkan lead time
    with col5:
        room_type_reserved = st.selectbox("Tipe Kamar Dipesan", options=room_type_options, index=0)
        avg_price_per_room = st.number_input("Harga Rata-rata per Kamar (€)", min_value=0.0, value=100.0, step=10.0, format="%.2f")
    with col6:
        market_segment_type = st.selectbox("Tipe Segmen Pasar", options=market_segment_options, index=0)
        no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, max_value=10, value=0, step=1)

    st.divider()
    st.subheader("Riwayat Tamu (jika ada)")
    col7, col8 = st.columns(2)
    with col7:
        no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0, step=1)
    with col8:
        no_of_previous_bookings_not_canceled = st.number_input("Jumlah Booking Sukses Sebelumnya", min_value=0, value=0, step=1)
        
    # Tombol Submit Form
    submitted = st.form_submit_button("Prediksi Status Booking")
    
    # --- Logika Setelah Form di-Submit ---
    if submitted:
        # Validasi input dasar (lead time tidak negatif)
        if lead_time < 0:
            st.error("Tanggal Kedatangan tidak boleh di masa lalu.")
        else:
            st.info("Memproses input dan melakukan prediksi...")

            # 1. Kumpulkan data dari input widget
            # Mapping Yes/No ke 1/0
            required_car_parking_space = 1 if required_car_parking_space_input == "Ya" else 0
            repeated_guest = 1 if repeated_guest_input == "Ya" else 0

            # Ekstrak tahun, bulan, tanggal
            arrival_year = arrival_date_input.year
            arrival_month = arrival_date_input.month
            arrival_date = arrival_date_input.day

        # 2. Dictionary Input (sesuai kolom SEBELUM preprocessing)
        input_dict = {
            'no_of_adults': [no_of_adults],
            'no_of_children': [no_of_children],
            'no_of_weekend_nights': [no_of_weekend_nights],
            'no_of_week_nights': [no_of_week_nights],
            'type_of_meal_plan': [type_of_meal_plan],
            'required_car_parking_space': [required_car_parking_space],
            'room_type_reserved': [room_type_reserved],
            'lead_time': [lead_time],
            'arrival_year': [arrival_year],
            'arrival_month': [arrival_month],
            'arrival_date': [arrival_date],
            'market_segment_type': [market_segment_type],
            'repeated_guest': [repeated_guest],
            'no_of_previous_cancellations': [no_of_previous_cancellations],
            'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
            'avg_price_per_room': [avg_price_per_room],
            'no_of_special_requests': [no_of_special_requests]
        }
        
        # 3. Konversi ke DataFrame Pandas
        input_df = pd.DataFrame.from_dict(input_dict)
        if 'expected_columns' in globals() and expected_columns is not None:
            pass 
        
        #4. Panggil Fungsi Prediksi dari inference.py
        predictions, probabilities = predict_booking_status(input_df)
        
        # 5. Tampilkan Hasil
        st.divider()
        st.subheader("Hasil Prediksi")
        if predictions is not None and probabilities is not None:
            prediction_result = predictions[0] 
            probability_canceled = probabilities[0]

            if prediction_result == 1:
                st.error(f"**Prediksi: Booking DIBATALKAN** (Probabilitas: {probability_canceled:.2%})", icon="🚫")
                st.warning("""
                Model memprediksi kemungkinan pembatalan yang tinggi untuk pemesanan ini.
                Pertimbangkan faktor risiko seperti lead time yang panjang, tamu baru, atau riwayat pembatalan.
                """)
            else:
                st.success(f"**Prediksi: Booking TIDAK Dibatalkan** (Probabilitas Pembatalan: {probability_canceled:.2%})", icon="✅")
                st.info("""
                Model memprediksi kemungkinan pembatalan yang rendah.
                """)

            # Tampilkan detail probabilitas
            st.metric(label="Probabilitas Pembatalan", value=f"{probability_canceled:.2%}")
        else:
            st.error("Gagal mendapatkan hasil prediksi. Silakan periksa log atau input Anda.")