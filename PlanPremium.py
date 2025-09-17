import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Cargar el dataset
data = pd.read_csv('dataset/PlanPremium.csv')

# 2. Preprocesamiento
X = data[['TiempoUsuarioMeses', 'ConsumoMensualGB', 'CategoriaDispositivo', 'PlanPremium']]
y = data['Contratacion']

# 3. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Estandarizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Predecir y matriz de confusión
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# 7. Graficar y guardar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Contratacion')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('static/confusion_matrix_planpremium.png')