import matplotlib.pyplot as rl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('student-mat.csv', sep=';')

# Ambil fitur dan target
X = df[['G1', 'G2']]
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2)
print("RMSE:", rmse)

# ============================
# Visualisasi Lebih Informatif + Nama
# ============================

rl.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Scatter plot prediksi vs aktual
sns.scatterplot(x=y_test, y=y_pred, color='blue', s=70, edgecolor='black', label='Data')

# Garis regresi
sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', label='Regresi Linear')

# Garis ideal (y = x)
rl.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2, label='Ideal (y = x)')

# Teks evaluasi di pojok kanan atas
rl.text(x=y_test.min(), y=y_test.max() - 2, 
         s=f'$R^2$: {r2:.2f}\nRMSE: {rmse:.2f}', 
         fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

# Label dan judul
rl.text(0.0, 1.05, 'Wahyu Agustino', 
         transform=rl.gca().transAxes, 
         fontsize=11, 
         ha='left', 
         va='center', 
         color='black')
rl.text(0.0, 1.01, 'Mata Kuliah Data Analys', 
         transform=rl.gca().transAxes,fontsize=11,
           ha='left', 
           va='center', 
           color='black')
rl.xlabel('Nilai G3 Aktual')
rl.ylabel('Nilai G3 Prediksi')
rl.title('Regresi Linear Nilai Akhir Siswa')
rl.legend()
rl.tight_layout()
rl.show()
