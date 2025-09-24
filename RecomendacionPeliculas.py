# RecomendacionPeliculas.py
# Caso Práctico - Recomendación de Películas
# Autor: César Bolívar y Angie Velasquez
# Descripción: Entrenamiento, evaluación y predicción de preferencia por películas de Acción

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 1. Configuración inicial
RANDOM_STATE = 42  # Semilla para reproducibilidad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "dataset_recomendacion_peliculas.csv")

print(f"📂 Cargando dataset desde: {DATA_PATH}")

# 2. Cargar y preparar datos
data = pd.read_csv(DATA_PATH)

# Objetivo binario: 1=Acción, 0=No Acción
data["Objetivo"] = np.where(data["GeneroPreferido"] == "Accion", 1, 0)

# Variables independientes
X = data[["EdadUsuario", "HistorialVisualizaciones",
          "PuntuacionesAnteriores", "GenerosFavoritos", "TiempoUsoMeses"]]
y = data["Objetivo"]

# 3. Split entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
# 4. Pipeline con escalado + modelo
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Normalización
    ("clf", LogisticRegression(random_state=RANDOM_STATE))  # Clasificador
])

# Entrenamiento
pipeline.fit(X_train, y_train)
# 5. Evaluación
def evaluate():
    """
    Evalúa el modelo entrenado en el conjunto de prueba.
    Devuelve métricas y guarda la matriz de confusión como imagen en /static.
    """
    y_pred = pipeline.predict(X_test)

    # Exactitud
    accuracy = accuracy_score(y_test, y_pred)

    # Reporte detallado
    report = classification_report(
        y_test, y_pred, target_names=["No-Accion", "Accion"], output_dict=True, zero_division=0
    )

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Gráfica
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    plt.title("Matriz de Confusión - Preferencia Acción")
    plt.xlabel("Predicho")
    plt.ylabel("Real")

    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", "confusion_matrix_peliculas.png")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Matriz de confusión guardada en: {output_path}")

    return {
        "accuracy": round(accuracy, 4),
        "report": report,
        "confusion_matrix": cm.tolist()
    }
# 6. Predicción para nuevos datos
def predict_label(features, threshold=0.5):
    """
    Recibe un diccionario con las features y devuelve:
    - Etiqueta ("Sí"/"No")
    - Probabilidad asociada
    Ejemplo:
    predict_label({
        "EdadUsuario": 25,
        "HistorialVisualizaciones": 100,
        "PuntuacionesAnteriores": 8,
        "GenerosFavoritos": 2,
        "TiempoUsoMeses": 12
    })
    """
    input_data = pd.DataFrame([features])
    prob = pipeline.predict_proba(input_data)[0][1]  # Prob de "Acción"
    label = "Sí" if prob >= threshold else "No"
    return {"label": label, "probabilidad": round(float(prob), 4)}



