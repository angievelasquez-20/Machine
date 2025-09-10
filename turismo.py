import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
import numpy as np

data = {
    'temp_media': [25, 28, 22, 30, 20, 27, 26, 23, 29, 21],
    'costo_pasaje': [500, 600, 450, 700, 400, 550, 520, 480, 650, 420],
    'num_turistas': [150, 120, 180, 90, 200, 140, 160, 170, 110, 190]
}

df = pd.DataFrame(data)
X = df[['temp_media', 'costo_pasaje']]
y = df['num_turistas']

model = LinearRegression()

model.fit(X, y)
def predict_tourists(temp_media, costo_pasaje):
    result = model.predict([[temp_media, costo_pasaje]])[0]
    return result