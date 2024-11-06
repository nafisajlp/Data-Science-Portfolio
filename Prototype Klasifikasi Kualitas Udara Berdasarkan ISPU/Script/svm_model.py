import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Baca data
data = pd.read_csv('C:/Users/NAFISAH/Downloads/ISPU_DASHBOARD/Data_Indeks Standar Pencemar Udara (ISPU) di Provinsi DKI Jakarta 2023.csv')

# Cleansing data
data = data.drop_duplicates()
data = data.replace(['-', '---'], np.nan)
data = data.dropna()

# Normalisasi
scaler = StandardScaler()
X = data[['pm_sepuluh', 'pm_duakomalima']]
X_scaled = scaler.fit_transform(X)

# Menjaga nama kolom setelah normalisasi
X_scaled_df = pd.DataFrame(X_scaled, columns=['pm_sepuluh', 'pm_duakomalima'])

# Penghapusan Fitur yang Tidak Relevan
data.drop(columns=['periode_data', 'stasiun', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida', 'max', 'parameter_pencemar_kritis'], inplace=True)

# One-hot encoding
data = pd.get_dummies(data, columns=['kategori'])

# Pilih label yang relevan
y = data[['kategori_BAIK', 'kategori_SANGAT TIDAK SEHAT', 'kategori_SEDANG', 'kategori_TIDAK SEHAT']]

# Pemeriksaan Outlier dan Penghapusannya
z_scores = np.abs(stats.zscore(X_scaled))
outlier_threshold = 3
to_keep = (z_scores < outlier_threshold).all(axis=1)

X_outlier_free = X_scaled_df[to_keep]
y_outlier_free = y[to_keep]

common_index = X_outlier_free.index.intersection(y_outlier_free.index)

X_final = X_outlier_free.loc[common_index]
y_final = y_outlier_free.loc[common_index]

# Pemisahan Data untuk Pelatihan dan Pengujian
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)

# Membangun model SVM dengan kernel linear
model = SVC(kernel='linear')
model.fit(X_train, y_train.values.argmax(axis=1))

# Simpan model dan scaler
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
