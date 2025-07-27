import pandas as pd
import matplotlib.pyplot as graph
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('student-mat.csv', sep=';')

# Ambil fitur yang relevan
features = df[['absences', 'studytime']]

# Normalisasi data agar clustering lebih akurat
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Tentukan jumlah klaster optimal (misalnya 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

cluster_names = {
    2: 'Rajin',
    1: 'Rata-rata',
    0: 'Kurang Rajin'
}
df['cluster_label'] = df['Cluster'].map(cluster_names)
print(df['cluster_label'].value_counts())

# Visualisasi hasil klaster
graph.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='absences', y='studytime', hue='Cluster', palette='viridis', s=70)
graph.title('Segmentasi Siswa Berdasarkan Absensi dan Waktu Belajar', fontsize=11)
graph.xlabel('Jumlah Absensi')
graph.ylabel('Waktu Belajar (1-4)')
graph.grid(True)

# Tambahkan nama dan kelas
graph.text(0.01, 1.05, 'Nama: Wahyu Agustino', transform=graph.gca().transAxes, fontsize=9, color='black')
graph.text(0.01, 1.01, 'Mata Kuliah: Data Analyst', transform=graph.gca().transAxes, fontsize=9, color='black')

graph.tight_layout()
graph.show()

