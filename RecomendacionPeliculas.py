# RecomendacionPeliculas.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===========================
# 1. Cargar datos
# ===========================
# ===========================
# 1. Cargar datos
# ===========================
# Ruta base donde está este archivo (RecomendacionPeliculas.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ahora apuntamos a la subcarpeta 'dataset'
DATA_PATH = os.path.join(BASE_DIR, "dataset", "dataset_recomendacion_peliculas.csv")

print("📂 Cargando dataset desde:", DATA_PATH)

# Leer el dataset
data = pd.read_csv(DATA_PATH)



# Definimos objetivo binario: Acción (1) vs No Acción (0)
data["Objetivo"] = np.where(data["GeneroPreferido"] == "Accion", 1, 0)

# Variables independientes
X = data[["EdadUsuario", "HistorialVisualizaciones",
          "PuntuacionesAnteriores", "GenerosFavoritos", "TiempoUsoMeses"]]

y = data["Objetivo"]

# ===========================
# 2. Split entrenamiento/prueba
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# 3. Pipeline con escalado y modelo
# ===========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42))
])

# ===========================
# 4. Entrenamiento
# ===========================
pipeline.fit(X_train, y_train)

# ===========================
# 5. Evaluación
# ===========================
def evaluate():
    """Entrena el modelo y devuelve métricas + guarda la matriz de confusión"""
    y_pred = pipeline.predict(X_test)

    # Exactitud
    accuracy = accuracy_score(y_test, y_pred)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=["No-Accion", "Accion"], output_dict=True)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Graficar matriz
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    plt.title("Matriz de Confusión - Preferencia Acción")
    plt.xlabel("Predicho")
    plt.ylabel("Real")

    os.makedirs("static", exist_ok=True)
    plt.savefig("static/confusion_matrix_peliculas.png")
    plt.close()

    return {
        "accuracy": round(accuracy, 4),
        "report": report,
        "confusion_matrix": cm.tolist()
    }

# ===========================
# 6. Predicción para nuevos datos
# ===========================
def predict_label(features, threshold=0.5):
    """
    Recibe un diccionario con las features y devuelve predicción "Sí"/"No" y probabilidad
    Ejemplo de features:
    {
        "EdadUsuario": 25,
        "HistorialVisualizaciones": 100,
        "PuntuacionesAnteriores": 8,
        "GenerosFavoritos": 2,
        "TiempoUsoMeses": 12
    }
    """
    input_data = pd.DataFrame([features])
    prob = pipeline.predict_proba(input_data)[0][1]  # probabilidad de ser "Acción"
    label = "Sí" if prob >= threshold else "No"
    return {"label": label, "probabilidad": round(prob, 4)}


