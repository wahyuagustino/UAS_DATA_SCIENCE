# Wahyu Agustino
# Mata Kuliah : Data Analyst

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as graph

# Load data
df = pd.read_csv('student-mat.csv', sep=';')

# Cek dimensi dan tipe data
print("Dimensi data:", df.shape)
print("\nTipe data dan non-null:\n", df.info())

# Statistik deskriptif
print("\nStatistik deskriptif numerik:\n", df.describe())

# Cek missing values
print("\nMissing values:\n", df.isnull().sum())

# Korelasi antar fitur numerik
graph.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
graph.title("Matriks Korelasi Antar Fitur Numerik")
graph.show()

# Distribusi nilai akhir (G3)
graph.figure(figsize=(8, 5))
sns.histplot(df['G3'], kde=True, bins=15, color='skyblue')
graph.title("Distribusi Nilai Akhir (G3)")
graph.xlabel("Nilai G3")
graph.ylabel("Jumlah Siswa")
graph.grid(True)
graph.show()

# Boxplot nilai akhir berdasarkan jenis kelamin
graph.figure(figsize=(6, 5))
sns.boxplot(x='sex', y='G3', data=df, palette='pastel')
graph.title("Distribusi Nilai G3 Berdasarkan Jenis Kelamin")
graph.xlabel("Jenis Kelamin")
graph.ylabel("Nilai Akhir (G3)")
graph.grid(True)
graph.show()

# Nilai rata-rata G3 berdasarkan studytime
graph.figure(figsize=(7, 5))
sns.barplot(x='studytime', y='G3', data=df, palette='Set2', estimator='mean')
graph.title("Rata-Rata Nilai Akhir Berdasarkan Waktu Belajar")
graph.xlabel("Waktu Belajar (1 (Rendah) - 4 (Tinggi))")
graph.ylabel("Rata-rata nilali G3")
graph.grid(True)
graph.show()

# Absensi vs G3
graph.figure(figsize=(8, 5))
sns.scatterplot(x='absences', y='G3', data=df, hue='sex', palette='Set1')
graph.title("Hubungan Absensi dan Nilai Akhir (G3)")
graph.xlabel("Jumlah Absensi")
graph.ylabel("Nilai Akhir (G3)")
graph.grid(True)
graph.show()
