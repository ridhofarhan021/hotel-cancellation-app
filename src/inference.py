import pickle
import pandas as pd
import numpy as np
import os

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model_final.pkl') # Gunakan model final dari tuning
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler_oop.pkl') # Sesuaikan jika nama file scaler berbeda
COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'columns.pkl')

try:
  with open(MODEL_PATH, 'rb') as f:
      model = pickle.load(f)
  print(f"Model berhasil dimuat dari {MODEL_PATH}")
except FileNotFoundError:
  print(f"Error: File model tidak ditemukan di {MODEL_PATH}")
  model = None
except Exception as e:
  print(f"Error saat memuat model: {e}")
  model = None

try:
  with open(SCALER_PATH, 'rb') as f:
      scaler = pickle.load(f)
  print(f"Scaler berhasil dimuat dari {SCALER_PATH}")
except FileNotFoundError:
  print(f"Error: File scaler tidak ditemukan di {SCALER_PATH}")
  scaler = None
except Exception as e:
  print(f"Error saat memuat scaler: {e}")
  scaler = None
    
try:
  with open(COLUMNS_PATH, 'rb') as f:
      expected_columns = pickle.load(f)
  print(f"Daftar kolom ({len(expected_columns)} kolom) berhasil dimuat dari {COLUMNS_PATH}")
except FileNotFoundError:
  print(f"Error: File daftar kolom tidak ditemukan di {COLUMNS_PATH}")
  expected_columns = None
except Exception as e:
  print(f"Error saat memuat daftar kolom: {e}")
  expected_columns = None

# --- Fungsi Prediksi : Melakukan prediksi status pembatalan booking berdasarkan data input.
def predict_booking_status(input_data):
  """
    Args:
        input_data (pd.DataFrame): DataFrame Pandas berisi satu atau lebih baris
                                    data booking baru. Kolom harus cocok dengan
                                    data asli SEBELUM encoding/scaling, tapi
                                    SETELAH feature engineering dasar jika ada
                                    (misal, total_nights, total_guests mungkin
                                    perlu dibuat jika tidak ada di input).

    Returns:
        tuple: Berisi (predictions, probabilities) atau (None, None) jika error.
              - predictions (np.array): Array berisi prediksi (0 atau 1).
              - probabilities (np.array): Array berisi probabilitas pembatalan (kelas 1).
    """
  if model is None or scaler is None or expected_columns is None:
          print("Error: Model, scaler, atau daftar kolom tidak berhasil dimuat. Prediksi dibatalkan.")
          return None, None
  if not isinstance(input_data, pd.DataFrame):
      print("Error: Input data harus berupa pandas DataFrame.")
      return None, None
    
  print(f"\nMenerima {input_data.shape[0]} baris data untuk prediksi.")
  # Buat salinan agar tidak mengubah DataFrame asli
  data = input_data.copy()
  try:
    # --- Tahap 1: Preprocessing Konsisten (Sebelum Scaling) ---
    print("  - Memulai preprocessing data input...")

    # 1. Feature Engineering 
    if 'total_nights' not in data.columns and 'no_of_weekend_nights' in data.columns and 'no_of_week_nights' in data.columns:
        data['total_nights'] = data['no_of_weekend_nights'] + data['no_of_week_nights']
        print("    - Fitur 'total_nights' dibuat.")
    
    if 'total_guests' not in data.columns and 'no_of_adults' in data.columns and 'no_of_children' in data.columns:
        data['total_guests'] = data['no_of_adults'] + data['no_of_children']
        print("    - Fitur 'total_guests' dibuat.")
    
    # Validasi 0 tamu 
    if (data['total_guests'] == 0).any():
      print("    - Warning: Terdapat input data dengan 0 tamu.")

    # 2. Koreksi Tipe Data 
    if 'required_car_parking_space' in data.columns:
      data['required_car_parking_space'].fillna(0, inplace=True)
      data['required_car_parking_space'] = data['required_car_parking_space'].astype(int)

    # 3. Encoding Fitur Kategorikal (One-Hot Encoding)
    categorical_cols_to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    valid_categorical_cols = [col for col in categorical_cols_to_encode if col in data.columns]

    if valid_categorical_cols:
      print(f"    - Melakukan One-Hot Encoding untuk: {valid_categorical_cols}")
      data = pd.get_dummies(data, columns=valid_categorical_cols, drop_first=True, dtype=int)
    else:
      print("    - Tidak ada kolom kategorikal valid yang ditemukan untuk OHE.")

    # --- Tahap 2: Penyesuaian Kolom & Scaling ---

    # 4. Menyamakan Kolom dengan Data Training 
    print(f"    - Menyamakan kolom dengan {len(expected_columns)} kolom training...")
    # Tambahkan kolom yang hilang (dari training) dengan nilai 0
    # Hapus kolom yang ada di input tapi tidak ada saat training
    data_reindexed = data.reindex(columns=expected_columns, fill_value=0)
    print(f"    - Ukuran data setelah reindex: {data_reindexed.shape}")


    # 5. Scaling Fitur Numerik (Menggunakan scaler yang sudah di-load)
    numerical_cols = data_reindexed.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [col for col in numerical_cols if data_reindexed[col].nunique(dropna=False) <= 2] # dropna=False penting jika ada NaN setelah reindex (seharusnya tidak jika fill_value=0)
    cols_to_scale = [col for col in numerical_cols if col not in binary_cols and col in scaler.feature_names_in_] # Pastikan kolom ada di scaler

    if not cols_to_scale:
      print("    - Tidak ada kolom yang perlu di-scale.")
    elif scaler is None: # jika scaler tidak dimuat
      print("    - Warning: Scaler tidak tersedia, scaling dilewati.")
    else:
      print(f"    - Melakukan scaling pada {len(cols_to_scale)} kolom...")
    
    # pengecek jika kolom ada di DataFrame sebelum scaling
    missing_in_df = [col for col in cols_to_scale if col not in data_reindexed.columns]
    if missing_in_df:
      print(f"    - Warning: Kolom {missing_in_df} tidak ditemukan di data input, tidak bisa di-scale.")
      cols_to_scale = [col for col in cols_to_scale if col in data_reindexed.columns]
    
    if cols_to_scale: # Jika masih ada kolom valid untuk di-scale
      data_reindexed[cols_to_scale] = scaler.transform(data_reindexed[cols_to_scale])
      print("    - Scaling selesai.")
    else:
      print("    - Tidak ada kolom valid yang tersisa untuk di-scale.")

    # --- Tahap 3: Prediksi ---
    print("  - Melakukan prediksi...")
    predictions = model.predict(data_reindexed)
    probabilities = model.predict_proba(data_reindexed)[:, 1] # Probabilitas kelas 1 (Canceled)
    print("Prediksi selesai.")
    return predictions, probabilities

  except Exception as e:
      print(f"Error selama proses prediksi: {e}")
      import traceback
      traceback.print_exc() # Cetak traceback untuk debug
      return None, None
  # --- Blok Pengujian (Contoh Penggunaan) ---
  
if __name__ == "__main__":
  print("\nMenjalankan pengujian fungsi prediksi...")

  contoh_data = pd.DataFrame({
      'no_of_adults': [2, 1],
      'no_of_children': [0, 1],
      'no_of_weekend_nights': [1, 0],
      'no_of_week_nights': [2, 3],
      'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2'], # Contoh nilai kategorikal
      'required_car_parking_space': [0, 1],
      'room_type_reserved': ['Room_Type 1', 'Room_Type 4'], # Contoh nilai kategorikal
      'lead_time': [50, 120],
      'arrival_year': [2025, 2025],
      'arrival_month': [5, 10],
      'arrival_date': [10, 25],
      'market_segment_type': ['Online', 'Offline'], # Contoh nilai kategorikal
      'repeated_guest': [0, 0],
      'no_of_previous_cancellations': [0, 1],
      'no_of_previous_bookings_not_canceled': [2, 0],
      'avg_price_per_room': [110.5, 95.0],
      'no_of_special_requests': [1, 0]
  })
  print("\nData Input Contoh:")
  print(contoh_data)

  # Panggil fungsi prediksi
  predictions, probabilities = predict_booking_status(contoh_data)

  # Tampilkan hasil
  if predictions is not None and probabilities is not None:
      print("\nHasil Prediksi:")
      hasil_df = pd.DataFrame({
          'Prediksi': predictions,
          'Status Prediksi': ['Canceled' if p == 1 else 'Not Canceled' for p in predictions],
          'Probabilitas Canceled': probabilities
      })
      print(hasil_df)
  else:
      print("\nPrediksi gagal.")

  print("\nPengujian selesai.")
