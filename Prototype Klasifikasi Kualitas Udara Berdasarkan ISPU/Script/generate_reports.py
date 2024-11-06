import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Baca data
data = pd.read_csv('C:/Users/NAFISAH/Downloads/result/datasets/Data_Indeks Standar Pencemar Udara (ISPU) di Provinsi DKI Jakarta 2023.csv')

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

# Periksa kolom kategori yang dihasilkan
print(data.columns)

# Pilih label yang relevan (sesuaikan jika kolom tidak ada)
kategori_columns = [col for col in data.columns if col.startswith('kategori_')]
y = data[kategori_columns]

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

# Prediksi
predictions = model.predict(X_scaled_df)

# Tambahkan kolom prediksi ke data asli
data['prediction'] = predictions

# Mapping kategori prediksi
category_mapping = {i: col for i, col in enumerate(kategori_columns)}
data['kategori_prediksi'] = data['prediction'].map(category_mapping)

# Mengkonversi kolom tanggal ke datetime dan set sebagai index
data['tanggal'] = pd.to_datetime(data['tanggal'])
data.set_index('tanggal', inplace=True)

# Pilih kolom yang akan digunakan untuk resampling (hanya kolom numerik)
data_numeric = data[['pm_sepuluh', 'pm_duakomalima', 'prediction']]

# Pastikan semua kolom numerik benar-benar tipe data numerik
data_numeric = data_numeric.apply(pd.to_numeric, errors='coerce')

# Fungsi untuk mendapatkan mode dengan aman
def get_mode(series):
    mode = series.mode()
    if len(mode) > 0:
        return mode[0]
    else:
        return np.nan

# Laporan Harian
daily_report = data_numeric.resample('D').mean()
daily_report['kategori_prediksi'] = data['kategori_prediksi'].resample('D').apply(get_mode)

# Laporan Mingguan
weekly_report = data_numeric.resample('W').mean()
weekly_report['kategori_prediksi'] = data['kategori_prediksi'].resample('W').apply(get_mode)

# Laporan Bulanan
monthly_report = data_numeric.resample('M').mean()
monthly_report['kategori_prediksi'] = data['kategori_prediksi'].resample('M').apply(get_mode)

# Laporan Triwulanan
quarterly_report = data_numeric.resample('Q').mean()
quarterly_report['kategori_prediksi'] = data['kategori_prediksi'].resample('Q').apply(get_mode)

# Laporan Semester
semester_report = data_numeric.resample('2Q').mean()
semester_report['kategori_prediksi'] = data['kategori_prediksi'].resample('2Q').apply(get_mode)

# Laporan Tahunan
yearly_report = data_numeric.resample('A').mean()
yearly_report['kategori_prediksi'] = data['kategori_prediksi'].resample('A').apply(get_mode)

# Simpan laporan jika diperlukan
daily_report.to_csv('daily_report.csv')
weekly_report.to_csv('weekly_report.csv')
monthly_report.to_csv('monthly_report.csv')
quarterly_report.to_csv('quarterly_report.csv')
semester_report.to_csv('semester_report.csv')
yearly_report.to_csv('yearly_report.csv')