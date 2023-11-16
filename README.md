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
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/5f2bf380-6b81-4eb7-98ce-3cce58cfbc4c" width=700px> </kbd> <br>
    Gambar 3 — Boxplot Numerical Feature
    </p>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/00d33684-9fea-4c0f-ad49-bd03a77c1781" width=700px> </kbd> <br>
    Gambar 4 — Barplot Categorical Feature
    </p>

- Terdapat outlier di fitur Area Income
- Clicked on Ad (fitur target) seimbang

## Bivariate Analysis
**Numerical Feature** <br>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/85d64250-38fe-4747-95f9-ff7d8c06b0b7" width=700px> </kbd> <br>
    Gambar 5 — Distribution Numerical Feature Based on Clicked Ads
    </p>
<p align="center">
    <kbd> <img src="https://github.com/arllarima/Predict-Clicked-Ads/assets/130117653/d4c2187c-73a4-4147-95ce-08ec22f3316d" width=700px> </kbd> <br>
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



