# 🌊 Prediksi Kecepatan Arus Laut dengan Long Short-Term Memory (LSTM)

Proyek *deep learning* untuk memprediksi **kecepatan arus laut (*velocity*)** berdasarkan data deret waktu dan koordinat lokasi menggunakan arsitektur **LSTM** bertingkat (*stacked LSTM*) dengan TensorFlow/Keras.

---

## 📌 Latar Belakang

Kecepatan arus laut penting untuk pelayaran, perikanan, dan kegiatan kelautan lainnya. Data arus bersifat **deret waktu** dan memiliki pola ketergantungan antarwaktu, sehingga LSTM dipilih karena mampu mempelajari pola jangka panjang pada data sekuensial.

## 📊 Dataset

| Keterangan | Detail |
|---|---|
| File | `velocity_2022.csv`, `velocity_2023.csv`, `velocity_2024.csv` |
| Periode | November 2022 – Juli 2024 |
| Kolom asli | `Time`, `Longitude`, `Latitude`, `Velocity` |
| Cakupan lokasi | 289 titik koordinat unik (sekitar 105,8° – 106,0° BT, 5,8° – 6,0° LS) |
| Jumlah data setelah penyaringan | ± 11,7 juta baris (sebelum menghapus duplikat) |
| Platform | Kaggle (dataset `arus-data`) |

Tidak ditemukan nilai kosong (*missing values*) maupun nilai tak hingga pada data. Sebanyak **181.431 baris duplikat** dihapus.

**Fitur yang digunakan:**

| Fitur | Keterangan |
|---|---|
| `Longitude`, `Latitude` | Koordinat lokasi |
| `Velocity` | Kecepatan arus |
| `year`, `month`, `day`, `hour` | Komponen waktu hasil ekstraksi dari kolom `Time` |

**Target:** `Velocity`

## ⚙️ Alur Pengerjaan

1. **Pengumpulan data**: menggabungkan data tahun 2022, 2023, dan 2024.
2. **Penyaringan lokasi**: mengambil 289 pasangan koordinat unik pertama.
3. **Ekstraksi fitur waktu**: kolom `Time` dipecah menjadi tahun, bulan, hari, dan jam.
4. **Pembersihan data**: pengecekan *missing values* dan penghapusan data duplikat.
5. **Normalisasi**: `MinMaxScaler` (rentang 0–1) pada fitur dan target.
6. **Pembentukan sekuens**: jendela waktu (*sliding window*) sepanjang **60 langkah** untuk memprediksi nilai berikutnya.
7. **Pembagian data**: 90% data latih dan 10% data uji (berurutan, tanpa diacak).
8. **Pelatihan model**: LSTM bertingkat dengan TPU (Kaggle).
9. **Evaluasi**: MSE, MAE, dan R² pada data uji, serta plot nilai aktual vs prediksi.

## 🧠 Arsitektur Model

```
Input (60 langkah × 7 fitur)
 ├─ LSTM(256, return_sequences=True) → Dropout(0.2)
 ├─ LSTM(256, return_sequences=True) → Dropout(0.2)
 ├─ LSTM(128, return_sequences=True) → Dropout(0.2)
 ├─ LSTM(128, return_sequences=True) → Dropout(0.2)
 ├─ LSTM(128)                        → Dropout(0.2)
 └─ Dense(1)
```

| Konfigurasi | Nilai |
|---|---|
| Optimizer | Adam (`learning_rate=0.0001`, `clipnorm=1.0`) |
| Loss | Mean Squared Error |
| Panjang sekuens | 60 |
| Batch size | 512 × jumlah replika TPU (8 core) |
| Epoch maksimum | 100 |
| Callback | `ModelCheckpoint` (menyimpan model terbaik berdasarkan `val_loss`) dan `EarlyStopping` (`patience=5`, `restore_best_weights=True`) |

## 📈 Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| **MSE** | **186,71** |
| **MAE** | **9,06** |
| **R² Score** | **0,8686** |

Model mampu menjelaskan sekitar **86,9%** variasi kecepatan arus pada data uji. Contoh perbandingan nilai aktual dan prediksi pada data uji:

| Real Velocity | Predicted Velocity |
|---|---|
| 101,92 | 102,21 |
| 93,30 | 100,42 |
| 107,06 | 96,39 |
| 109,32 | 102,77 |
| 103,08 | 101,09 |

## 📁 Struktur Proyek

```
long-short-term-memory/
├── arus_pred_86.ipynb     # Notebook: preprocessing, pelatihan LSTM, evaluasi
├── model_arus.keras       # Model LSTM terlatih (jika disertakan)
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/angfdzr/long-short-term-memory.git
cd long-short-term-memory
```

### 2. Install dependensi

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib jupyter
```

### 3. Siapkan dataset

Letakkan file `velocity_2022.csv`, `velocity_2023.csv`, dan `velocity_2024.csv` di komputer Anda, lalu ubah path pada sel pembacaan data di notebook (semula `/kaggle/input/arus-data/...`).

### 4. Jalankan notebook

```bash
jupyter notebook arus_pred_86.ipynb
```

> **Catatan:** Notebook ini awalnya dijalankan di **Kaggle dengan TPU**. Satu epoch memakan waktu sekitar 6 menit di TPU. Jika dijalankan di CPU/GPU lokal, kode otomatis memakai strategi default, namun pelatihan akan jauh lebih lama. Path model (`/kaggle/working/...` dan `/kaggle/input/...`) juga perlu disesuaikan.

## 🛠️ Teknologi yang Digunakan

- **Python 3.10**
- **TensorFlow / Keras**: model LSTM
- **pandas** & **NumPy**: pengolahan data
- **scikit-learn**: `MinMaxScaler` dan metrik evaluasi
- **matplotlib**: visualisasi
- **Kaggle TPU**: akselerasi pelatihan

## ⚠️ Keterbatasan & Pengembangan Lanjutan

- `Velocity` dipakai sebagai fitur sekaligus target, sehingga model sebagian besar belajar dari nilai arus sebelumnya.
- Sekuens dibentuk dari baris data berurutan, bukan per titik koordinat, sehingga satu jendela 60 langkah dapat memuat beberapa lokasi berbeda.
- Pembagian data latih/uji berurutan sehingga data uji hanya mewakili periode paling akhir.
- Pengembangan lanjutan: membentuk sekuens per lokasi, mencoba fitur siklik untuk waktu (sin/cos), serta membandingkan dengan model lain seperti GRU atau Random Forest.

## 👤 Penulis

**Angga Fadzar**
Mahasiswa

---

*Proyek ini dibuat untuk keperluan pembelajaran dan penelitian.*
