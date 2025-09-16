import io
from xml.parsers.expat import model
from flask import Flask
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from turismo import predict_tourists
import RegresionLineal as RegresionLineal
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

app = Flask(__name__)

@app.route('/')
def Index():
     Myname= "Flask"
     return render_template('Index.html', name=Myname)

@app.route('/Caso1')
def caso():     
     return render_template('Caso1.html')

@app.route('/Caso2')
def caso2():     
     return render_template('Caso2.html')

@app.route('/Caso3')
def caso3():
     return render_template('Caso3.html')

@app.route('/Caso4')
def caso4():
     return render_template('Caso4.html')

@app.route('/LR', methods=['GET','POST'])
def LR():
     calculateResult = None
     if request.method == 'POST':
          hours = float(request.form['hours'])
          calculateResult = RegresionLineal.calculateGrade(hours)
     return render_template("templateRegresion.html", result = calculateResult)

@app.route('/Actividad4', methods=['GET', 'POST'])
def actividad4():
    if request.method == 'POST':
        try:
            temp_media = float(request.form['temp_media'])
            costo_pasaje = float(request.form['costo_pasaje'])
            turistas_estimados = predict_tourists(temp_media, costo_pasaje)
            return jsonify({
                'success': True,
                'turistas_estimados': round(turistas_estimados, 2)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error en la predicción: {str(e)}'
            })
    return render_template('Actividad4.html')

@app.route('/confusion_matrix')
def confusion_matrix():
    return render_template('confusion_matrix.html')
          

if __name__ == '__main__':
     app.run(debug=True)