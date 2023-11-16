# Predict Clicked Ads Customer Classification by Using Machine Learning
**Dataset** : Provided by Rakamin Academy <br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, Matplotlib, Seaborn, Scikit Learn

# Overview
Sebuah perusahaan di Indonesia ingin mengetahui efektifitas sebuah iklan yang mereka tayangkan, hal ini penting bagi perusahaan agar dapat mengetahui seberapa besar ketercapainnya iklan yang dipasarkan sehingga dapat menarik customers untuk melihat iklan. <br>
Dengan mengolah data historical advertisement serta menemukan insight serta pola yang terjadi, maka dapat membantu perusahaan dalam menentukan target marketing, fokus case ini adalah membuat model machine learning classification yang berfungsi menentukan target customers yang tepat.

# Exploratory Data Analysis
- Dataset terdiri dari 1000 baris dan 11 fitur.
- Dataset memiliki 5 fitur numerikal dan 6 fitur kategorikal.

## Statistical Analysis
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/02b2fb2f-dd6f-4241-b33b-3f781ae1bf4e" width=700px> </kbd> <br>
    Gambar 1 — Numerical Feature
    </p>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/c37a7ff4-276f-4fbb-bf60-08f27521aa63" width=400px> </kbd> <br>
    Gambar 2 — Categorical Feature
    </p>

Numerical Feature:
- Rata-rata user menghabiskan waktu di suatu situs adalah 1 jam (65 menit).
- Rata-rata umur user adalah 36 tahun.
- Rata-rata pendapatan user adalah Rp 384.864.671 / tahun.
- Rata-rata pemakaian internet harian user adalah 3 jam (180 menit).

Categorical Feature:
- Kebanyakan user adalah perempuan.
- Clicked on Ad Yes dan No memiliki jumlah value yang seimbang.
- Surabaya adalah kota dengan user terbanyak.
- Otomotif adalah kategori yang paling sering di klik.

## Univariate Analysis
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/5f2bf380-6b81-4eb7-98ce-3cce58cfbc4c" width=600px> </kbd> <br>
    Gambar 3 — Boxplot Numerical Feature
    </p>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/00d33684-9fea-4c0f-ad49-bd03a77c1781" width=600px> </kbd> <br>
    Gambar 4 — Barplot Categorical Feature
    </p>

- Terdapat outlier di fitur Area Income
- Clicked on Ad (fitur target) seimbang

## Bivariate Analysis
**Numerical Feature** <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/85d64250-38fe-4747-95f9-ff7d8c06b0b7" width=600px> </kbd> <br>
    Gambar 5 — Distribution Numerical Feature Based on Clicked Ads
    </p>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/d4c2187c-73a4-4147-95ce-08ec22f3316d" width=600px> </kbd> <br>
    Gambar 6 — Distribution (Violin Plot) Numerical Feature Based on Clicked Ads
    </p>

**Daily Time Spent**
- User yang sedikit menghabiskan waktu di sebuah situs (kurang dari 1 jam) memiliki potensi untuk mengklik iklan yang lebih besar

**Daily Internet Usage**
- User dengan pemakaian internet yang rendah memiliki potensi untuk mengklik iklan yang lebih besar.

**Age**
- User yang lebih tua memiliki potensi mengklik iklan yang lebih besar.
<br>

**Categorical Feature**
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/b975ceb2-66a8-4f20-b8de-bba9d5474b8b" width=500px> </kbd> <br>
    Gambar 7 — Clicked on Ads Distribution by Gender
    </p>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/ff65cc95-576d-42bd-bde7-17f5e91d7551" width=500px> </kbd> <br>
    Gambar 8 — Clicked on Ads Distribution by Category
    </p>

- Secara keseluruhan perbedaan gender pada potensi klik iklan tidak terlalu signifikan
- Lebih banyak perempuan yang mengklik ads.
- Kategori dengan potensi klik tertinggi adalah Finance, Fashion, dan Otomotif.

<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/56d3e2f4-bca1-4a15-941c-5520fd89aab5" width=500px> </kbd> <br>
    Gambar 9 — Daily Total Clicked Ads
    </p>

- Banyak user yang mengklik iklan di hari Rabu, Kamis, dan Minggu.
- Hari Kamis memiliki konversi klik iklan yang paling baik, jumlah user yang mengklik tinggi dan yang tidak mengklik rendah.

<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/f2c81018-339c-4bfe-8304-aae6b6bad280" width=500px> </kbd> <br>
    Gambar 9 — Hourly Total Clicked Ads
    </p>

- Potensi user untuk mengklik iklan lebih tinggi pada pukul 00.00, 09.00, dan 18.00.
- Ini mungkin karena kebiasaan pengguna saat menggunakan perangkat digital.
- Pada pukul 00.00, orang mungkin tidak punya banyak tugas yang mendesak, jadi mereka lebih mungkin menghabiskan waktu online.
- Pukul 09.00 mungkin menjadi waktu jeda di antara pekerjaan atau istirahat singkat.
- Pukul 18.00 adalah waktu setelah bekerja di mana mereka bisa fokus pada kegiatan pribadi.

## Multivariate Analysis
**Heatmap Correlation**
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/2424a4ca-2937-4b46-a5bc-f82b423f8ebd" width=600px> </kbd> <br>
    Gambar 10 — Heatmap Correlation
    </p>

- Tidak ada fitur yang redundan.
- Korelasi tertinggi adalah antara Daily Time Spent on Site dengan Daily Internet Usage, dimana semakin lama user menghabiskan waktu di site maka semakin tinggi pemakaian internet hariannya.
- Semakin besar income user juga semakin besar pemakaian internet hariannya.

# Data Pre-Processing
**Table 1. Data Pre-Processing** <br>
**No**  |     **Treatment**      |    **Actions**     |
:-----: |    ----------------    |    ------------    |
1 |   Handling Missing Values    |   - Mengisi missing values dengan nilai modus di fitur `Male` <br> - Mengisi missing values dengan nilai median di fitur `Daily Time Spent on Site`, `Area Income`, `Daily Internet Usage`|
2 |   Change Datatype     |   - Mengganti datatype Timestamp menjadi datetime <br> - Mengekstrak Tangal, Hari, dan Waktu (Jam). |
3 |    Duplicated Data    | - Tidak ada data yang duplikat |
4 |  Feature Encoding | - Label Encoding: `Male`, `Clicked on Ad` <br> - One Hot Encoding: `Category` |
5 | Feature Extraction | - Menghapus fitur yang tidak digunakan dalam model, yaitu: `Unnamed:0`, `city`, `province` |

# Data Modeling
## Machine Learning
**Experiment 1 - Machine Learning Tanpa Standarisasi Fitur**
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/ca06dc5f-c5e4-4b0b-8a04-e7213c356f47" width=500px> </kbd> <br>
    Gambar 11 — Machine Learning Tanpa Standarisasi Fitur
    </p>

**Experiment 2 - Machine Learning Dengan Standarisasi Fitur**
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/2c455b90-470c-47e2-ad7a-9f10023dfad9" width=500px> </kbd> <br>
    Gambar 12 — Machine Learning Dengan Standarisasi Fitur
    </p>

- Hasil model tanpa standardisasi yang memiliki nilai akurasi tertinggi adalah Random Forest.
- Algoritma lain yang juga memiliki akurasi yang tinggi adalah XGBoost dan Decision Tree.
- Hasil model menggunakan standarisasi yang memiliki nilai akurasi tertinggi adalah Logistic Regression.
- Pada algoritma SVC dan Logistic Regression hasil akurasi meningkat secara signifikan dengan menggunakan standarisasi fitur.

## Model Evaluation
**Confusion Matrix**
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/485a6670-29fc-4a4e-b9bf-eb4839fc79e5" width=500px> </kbd> <br>
    Gambar 13 — Confusion Matrix
    </p>

- Berdasarkan evaluasi algoritma Logistic Regression dari confussion matrix terlihat bahwa model sangat baik memprediksi user yang klik iklan atau tidak dengan nilai kesalahan prediksi yang kecil.
- Terdapat 143 prediksi benar yang diklasifikasikan user yang tidak mengklik iklan (True Negatives, TN).
- Terdapat 3 prediksi salah yang diklasifikasikan sebagai user yang mengklik iklan padahal sebenarnya bukan (False Positives, FP).
- Terdapat 6 prediksi salah yang diklasifikasikan sebagai user yang tidak mengklik iklan padahal sebenarnya adalah mengklik iklan (False Negatives, FN).
- Terdapat 148 prediksi benar yang diklasifikasikan sebagai user yang mengklik iklan (True Positives, TP).

## Feature Importance
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/b3f0a283-4116-4d35-9052-026643665d19" width=500px> </kbd> <br>
    Gambar 14 — Feature Importance
    </p>

<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/c281faec-4023-4dc2-8a32-a6fa4d1a75c8" width=250px> </kbd> <br>
    Gambar 15 — Coefficient Feature Importance
    </p>

- Dari plot feature importance, dapat disimpulkan bahwa Daily Internet Usage, Daily Time Spent on Site, Area Income, dan Age adalah fitur yang paling berpengaruh dalam memprediksi klik iklan.
- Fitur-fitur tersebut memiliki koefisien magnitudo yang paling besar, dan berkorelasi negatif untuk Daily Internet Usage, Daily Time Spent on Site, dan Area Income, menunjukkan bahwa user yang tidak aktif online dan memiliki pendapatan menengah ke bawah cenderung lebih mungkin mengklik iklan.
- Sebaliknya, fitur Age memiliki korelasi positif, menandakan bahwa semakin tua usia user, semakin tinggi potensi untuk mengklik iklan.

# Business Recommendation
Rekomendasi berdasarkan Feature Importance dan insight, adalah: <br>

**1.   Target Pengguna Non-Aktif:**
- Buat iklan singkat dan menarik untuk menangkap perhatian pengguna yang jarang menghabiskan waktu di situs (kurang dari 1 jam) dan jarang menggunakan internet, karena pengguna ini cenderung lebih mungkin mengklik iklan.
- Manfaatkan retargeting dengan menampilkan iklan yang relevan berulang kali untuk meningkatkan awareness.

**2.   Relevansi Konten Iklan:**
- Pastikan konten iklan sesuai dengan minat dan kebutuhan pengguna non-aktif.

**3.   Penawaran Harga Terjangkau:**
- Berikan penawaran harga yang terjangkau, diskon, atau promo untuk menarik pengguna dengan area income rendah.

**4.   Optimalisasi Waktu dan Hari:**
- Manfaatkan hari Rabu, Kamis, dan Minggu untuk penayangan iklan karena memiliki konversi klik iklan yang baik.
- Pilih jam-jam pukul 00.00, 09.00, dan 18.00 yang menunjukkan potensi tinggi pengguna mengklik iklan dan melakukan pembelian.








    
