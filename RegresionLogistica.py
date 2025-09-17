import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

def evaluate_model(y_true, y_pred, model_name="Modelo de Clasificación"):
    print(f"Evaluación del {model_name}")
    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Exactitud del modelo: {accuracy * 100:.2f}%')
    return accuracy

data = pd.read_csv('data.cvs')

print(data.head())
print(data.info())
print(data.describe())

x=data.drop('HeartDisease', axis=1)
y=data['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

logistic_model = LogisticRegression()
logistic_model.fit(x_train_scaled, y_train)

y_pred = logistic_model.predict(x_test_scaled)

confusion_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('static/confusion_matrix.png')




print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'exactitud del modelo: {accuracy * 100:.2f}%')