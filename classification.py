# Wahyu Agustino
# Mata Kuliah Data Analyst
# Klasifikasi berdasarkan 3 variabel ( bebas )

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("student-mat.csv", sep=';')

# Buat label biner: Lulus (1) atau Tidak (0)
df['target'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Pilih fitur
X = df[['studytime', 'absences', 'failures']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model klasifikasi
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualisasi feature importance
plt.figure(figsize=(6,4))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Klasifikasi 3 variabel ( bebas )")
plt.show()
