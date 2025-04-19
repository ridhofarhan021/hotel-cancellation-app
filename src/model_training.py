import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

class HotelBookingTrainer:
  """
  Attributes:
    data_path (str): Path ke file CSV dataset.
    model_save_path (str): Path untuk menyimpan model terlatih (.pkl).
    scaler_save_path (str): Path untuk menyimpan objek scaler (.pkl).
    best_params (dict): Dictionary berisi hyperparameter terbaik untuk model.
    target_column (str): Nama kolom target.
    test_size (float): Ukuran proporsi data test.
    random_state (int): Random state untuk reproduktivitas.
  """
  def __init__(self, data_path, model_save_path, scaler_save_path, columns_save_path, best_params, # Tambahkan columns_save_path
            target_column='booking_status', test_size=0.2, random_state=42):
    """Inisialisasi HotelBookingTrainer."""
    self.data_path = data_path
    self.model_save_path = model_save_path
    self.scaler_save_path = scaler_save_path
    self.best_params = best_params
    self.target_column = target_column
    self.test_size = test_size
    self.random_state = random_state
    self.scaler = None # Akan diinisialisasi saat scaling
    self.trained_model = None # Akan diisi setelah training
    self.columns_save_path = columns_save_path # Simpan path kolom
    
    os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(self.columns_save_path), exist_ok=True) 
    
    print("HotelBookingTrainer diinisialisasi.")
    print(f"  - Data Path: {self.data_path}")
    print(f"  - Model Save Path: {self.model_save_path}")
    print(f"  - Scaler Save Path: {self.scaler_save_path}")
    print(f"  - Best Params: {self.best_params}")
    print(f"  - Columns Save Path: {self.columns_save_path}")
    
    
  def load_data(self):
    """Memuat data dari path CSV."""
    print(f"\nMemuat data dari {self.data_path}...")
    try:
      df = pd.read_csv(self.data_path)
      print("Data berhasil dimuat.")
      return df
    except FileNotFoundError:
      print(f"Error: File tidak ditemukan di {self.data_path}")
      return None
    
  def preprocess_data(self, df):
    print("\nMemulai Preprocessing Data...")
    if df is None:
      return None
    
    start_time = time.time()

    # 1. Imputasi Missing Values (sesuai strategi dari notebook)
    print("  - Menangani missing values...")
    if 'avg_price_per_room' in df.columns:
      median_price = df['avg_price_per_room'].median()
      df['avg_price_per_room'].fillna(median_price, inplace=True)
    if 'required_car_parking_space' in df.columns:
      df['required_car_parking_space'].fillna(0, inplace=True)
    if 'type_of_meal_plan' in df.columns:
      mode_meal = df['type_of_meal_plan'].mode()[0]
      df['type_of_meal_plan'].fillna(mode_meal, inplace=True)
      
    # 2. Koreksi Tipe Data
    print("  - Mengoreksi tipe data...")
    if 'required_car_parking_space' in df.columns:
      df['required_car_parking_space'] = df['required_car_parking_space'].astype(int)
    
    # 3. Hapus Duplikat
    print("  - Menghapus data duplikat...")
    num_duplicates_before = df.duplicated().sum()
    if num_duplicates_before > 0:
      df.drop_duplicates(inplace=True)
      print(f"    {num_duplicates_before} baris duplikat dihapus.")

    # 4. Feature Engineering
    print("  - Melakukan feature engineering...")
    if 'no_of_weekend_nights' in df.columns and 'no_of_week_nights' in df.columns:
      df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    if 'no_of_adults' in df.columns and 'no_of_children' in df.columns:
      df['total_guests'] = df['no_of_adults'] + df['no_of_children']
      # Hapus baris dengan 0 tamu
      initial_rows = df.shape[0]
      df = df[df['total_guests'] > 0]
      if df.shape[0] < initial_rows:
        print(f"    {initial_rows - df.shape[0]} baris dengan 0 tamu dihapus.")

    # 5. Encoding Target Variable
    print("  - Encoding target variable...")
    if self.target_column in df.columns:
      target_map = {'Canceled': 1, 'Not_Canceled': 0}
      df[self.target_column] = df[self.target_column].map(target_map)
    else:
      print(f"Error: Kolom target '{self.target_column}' tidak ditemukan.")
      return None
      
    # 6. Encoding Fitur Kategorikal (One-Hot Encoding)
    print("  - Encoding fitur kategorikal...")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'Booking_ID' in categorical_cols:
      categorical_cols.remove('Booking_ID') # tidak melakukan encode ID

    expected_cats = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    valid_categorical_cols = [col for col in categorical_cols if col in expected_cats]
    print(f"    Kolom yang akan di OHE: {valid_categorical_cols}")

    if valid_categorical_cols:
      df = pd.get_dummies(df, columns=valid_categorical_cols, drop_first=True, dtype=int)

    # Hapus kolom ID jika masih ada
    if 'Booking_ID' in df.columns:
      df.drop('Booking_ID', axis=1, inplace=True)

    end_time = time.time()
    print(f"Preprocessing selesai dalam {end_time - start_time:.2f} detik.")
    print(f"Ukuran data setelah preprocess: {df.shape}")
    return df
  
  def split_and_scale_data(self, df):
    print("\nMemisahkan dan men-scale data...")
    if df is None or self.target_column not in df.columns:
      print("Error: Data tidak valid atau kolom target tidak ada.")
      return None

    try:
      X = df.drop(self.target_column, axis=1)
      y = df[self.target_column]
      
      # Simpan Kolom Fitur (untuk Inference)
      feature_columns = X.columns.tolist()
      try:
        with open(self.columns_save_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"  - Daftar kolom fitur ({len(feature_columns)} kolom) disimpan ke: {self.columns_save_path}")
      except Exception as e:
        print(f"  - Warning: Gagal menyimpan daftar kolom: {e}")
      
      # Pemisahan Data
      X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=self.test_size,
        random_state=self.random_state,
        stratify=y
      )
      print(f"  - Data dibagi: Train ({X_train.shape}), Test ({X_test.shape})")

      # Identifikasi kolom numerik untuk scaling (non-biner)
      numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
      binary_cols = [col for col in numerical_cols if X_train[col].nunique() <= 2]
      cols_to_scale = [col for col in numerical_cols if col not in binary_cols]
      print(f"  - Kolom yang akan di-scale ({len(cols_to_scale)} kolom)") #: {cols_to_scale}")

      if not cols_to_scale:
        print("  - Tidak ada kolom yang perlu di-scale.")
        # Simpan scaler 'kosong' 
        self.scaler = None 
        return X_train, X_test, y_train, y_test
      
      else:
          # Scaling
          self.scaler = StandardScaler()
          X_train[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
          X_test[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
          print("  - Fitur numerik telah di-scale.")

          # Simpan scaler
          with open(self.scaler_save_path, 'wb') as f:
              pickle.dump(self.scaler, f)
          print(f"  - Scaler disimpan ke: {self.scaler_save_path}")

          return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error saat split/scale data: {e}")
        return None

  def train_model(self, X_train, y_train):
    """
    Args:
        X_train (pd.DataFrame): Fitur training yang sudah di-scale.
        y_train (pd.Series): Target training.

    Returns:
        object: Model yang sudah dilatih atau None jika error.
    """
    print("\nMemulai Pelatihan Model Terbaik...")
    if X_train is None or y_train is None:
          print("Error: Data training tidak valid.")
          return None
    if not self.best_params:
          print("Error: Hyperparameter terbaik (`best_params`) tidak disediakan.")
          return None
    start_time = time.time()
    try:
      model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            **self.best_params # Menggunakan parameter terbaik
        )
      # model = RandomForestClassifier(
      #     random_state=self.random_state,
      #     n_jobs=-1,
      #     **self.best_params
      # )
      # ---------------------------------------------
      
      # Latih model
      model.fit(X_train, y_train)
      self.trained_model = model # Simpan model terlatih ke atribut kelas

      end_time = time.time()
      print(f"Pelatihan model selesai dalam {end_time - start_time:.2f} detik.")

      # Simpan model terlatih
      with open(self.model_save_path, 'wb') as f:
          pickle.dump(self.trained_model, f)
      print(f"Model terlatih disimpan ke: {self.model_save_path}")

      return self.trained_model
    
    except Exception as e:
          print(f"Error saat melatih/menyimpan model: {e}")
          return None
    
  def evaluate_model(self, X_test, y_test):
        """
        Mengevaluasi model terlatih pada data test.

        Args:
            X_test (pd.DataFrame): Fitur test yang sudah di-scale.
            y_test (pd.Series): Target test.
        """
        print("\nMengevaluasi Model pada Data Test...")
        if self.trained_model is None:
          print("Error: Model belum dilatih.")
          return
        if X_test is None or y_test is None:
          print("Error: Data test tidak valid.")
          return

        try:
          y_pred = self.trained_model.predict(X_test)
          y_prob = self.trained_model.predict_proba(X_test)[:, 1] # Prob untuk kelas 1

          print("\nLaporan Klasifikasi (Test Set):")
          print(classification_report(y_test, y_pred, target_names=['Not Canceled (0)', 'Canceled (1)']))

          roc_auc = roc_auc_score(y_test, y_prob)
          print(f"ROC AUC Score (Test Set): {roc_auc:.4f}")

        except Exception as e:
          print(f"Error saat mengevaluasi model: {e}")

  def run_training_pipeline(self):
    """Menjalankan seluruh pipeline training."""
    print("="*50)
    print("Memulai Training Pipeline")
    print("="*50)

    # 1. Load Data
    df_raw = self.load_data()
    if df_raw is None:
        print("Pipeline dihentikan karena gagal memuat data.")
        return

    # 2. Preprocess Data
    df_processed = self.preprocess_data(df_raw.copy()) # Gunakan copy agar tidak mengubah df_raw asli
    if df_processed is None:
        print("Pipeline dihentikan karena gagal preprocess data.")
        return

    # 3. Split and Scale Data
    split_result = self.split_and_scale_data(df_processed)
    if split_result is None:
        print("Pipeline dihentikan karena gagal split/scale data.")
        return
    X_train_scaled, X_test_scaled, y_train, y_test = split_result

    # 4. Train Model
    trained_model = self.train_model(X_train_scaled, y_train)
    if trained_model is None:
        print("Pipeline dihentikan karena gagal melatih model.")
        return

    # 5. Evaluate Model
    self.evaluate_model(X_test_scaled, y_test)

    print("\nTraining Pipeline Selesai.")
    print("="*50)
  
if __name__ == "__main__":
  print("Menjalankan skrip model_training.py...")

  # --- Konfigurasi Path ---
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Mendapatkan direktori induk dari src
  DATA_PATH = os.path.join(BASE_DIR, 'Dataset_B_hotel.csv')
  MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'best_model_oop.pkl') # Nama model baru
  SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'scaler_oop.pkl')
  COLUMNS_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'columns.pkl') # Path untuk kolom

  # Nama scaler baru
  BEST_MODEL_PARAMS = {
      'subsample': 0.8,           
      'n_estimators': 200,        
      'max_depth': 9,             
      'learning_rate': 0.1,       
      'gamma': 0.2,               
      'colsample_bytree': 0.7,     
      'reg_lambda': 1,
      'reg_alpha': 0
  }
  # --- Instance Trainer dan Jalankan Pipeline ---
  trainer = HotelBookingTrainer(
    data_path=DATA_PATH,
    model_save_path=MODEL_SAVE_PATH,
    scaler_save_path=SCALER_SAVE_PATH,
    columns_save_path=COLUMNS_SAVE_PATH, 
    best_params=BEST_MODEL_PARAMS
)
  trainer.run_training_pipeline()
  
  print("\nEksekusi skrip model_training.py selesai.")