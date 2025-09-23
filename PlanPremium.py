# RecomendacionPeliculas.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Semilla global para reproducibilidad
RANDOM_STATE = 42

# ================================
# 1. CARGA DE DATOS
# ================================
# Cargar dataset
data = pd.read_csv("dataset_recomendacion_peliculas.csv")

# Variables independientes
X = data[['EdadUsuario', 'HistorialVisualizaciones', 'PuntuacionesAnteriores', 
          'GenerosFavoritos', 'TiempoUsoMeses']]

# Variable objetivo binaria: Romance = 1, otros = 0
y = (data['GeneroPreferido'] == "Romance").astype(int)

# Documentación:
# Clase 1 = Sí (Romance preferido)
# Clase 0 = No (otros géneros)

# ================================
# 2. SPLIT TRAIN/TEST
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ================================
# 3. PIPELINE DE PREPROCESAMIENTO + MODELO
# ================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Escalado
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE))  # Algoritmo
])

# ================================
# 4. ENTRENAMIENTO
# ================================
pipeline.fit(X_train, y_train)

# ================================
# 5. EVALUACIÓN
# ================================
def evaluate():
    """Evalúa el modelo entrenado y guarda matriz de confusión"""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No", "Sí"], output_dict=True)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Graficar
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    plt.title("Matriz de Confusión - Romance vs No Romance")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/confusion_matrix_peliculas.png")

    return {
        "accuracy": round(acc, 4),
        "precision": round(report["Sí"]["precision"], 4),
        "recall": round(report["Sí"]["recall"], 4),
        "f1": round(report["Sí"]["f1-score"], 4)
    }

# ================================
# 6. PREDICCIÓN INDIVIDUAL
# ================================
def predict_label(features, threshold=0.5):
    """
    features: lista o array con [edad, historial, puntuacion, genero_fav, tiempo]
    Retorna: ("Sí"/"No", probabilidad)
    """
    proba = pipeline.predict_proba([features])[0][1]
    label = "Sí" if proba >= threshold else "No"
    return label, round(proba, 4)

# ================================
# 7. GUARDAR MODELO
# ================================
with open("modelo_peliculas.pkl", "wb") as f:
    pickle.dump(pipeline, f)

if __name__ == "__main__":
    metrics = evaluate()
    print("✅ Evaluación del modelo:")
    print(metrics)
    ejemplo = [25, 120, 8, 5, 12]  # ejemplo de features
    print("🔮 Predicción ejemplo:", predict_label(ejemplo))
