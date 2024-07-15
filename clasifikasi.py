# Import library yang diperlukan
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Memuat data polusi udara (contoh: data dapat diambil dari sumber yang tersedia)
# Misalnya, load data dari file CSV
# Ubah dengan path file yang sesuai
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Memilih fitur dan target
X = df[['PM10', 'PM25', 'CO', 'HC', 'O3', 'SO2', 'NO2']]
y = df['ISPU']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline untuk preprocessing dan model SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalisasi fitur
    ('svm', SVC(kernel='rbf'))     # SVM dengan kernel RBF
])

# Melatih model SVM
pipeline.fit(X_train, y_train)

# Menguji model pada data uji
y_pred = pipeline.predict(X_test)

# Evaluasi hasil klasifikasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi klasifikasi: {accuracy:.2f}')

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred))

# Menentukan kategori ISPU berdasarkan hasil prediksi
def kategori_ispu(ispu):
    if ispu <= 50:
        return 'Baik'
    elif ispu <= 100:
        return 'Sedang'
    elif ispu <= 200:
        return 'Tidak Sehat'
    elif ispu <= 300:
        return 'Sangat Tidak Sehat'
    else:
        return 'Berbahaya'

# Contoh penggunaan
contoh_data = np.array([[65, 30, 0.5, 0.2, 0.1, 0.03, 0.04]])  # Ganti dengan data yang sesuai
prediksi_ispu = pipeline.predict(contoh_data)
kategori = kategori_ispu(prediksi_ispu[0])
print(f'Prediksi ISPU: {prediksi_ispu[0]}, Kategori: {kategori}')
