from xml.parsers.expat import model
from flask import Flask
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    name = None
    name = "Flask"
    return f"hello, {name}!"

@app.route('/Index')
def Index():
     Myname= "Flask"
     return render_template('Index.html', name=Myname)

@app.route('/Actividad3')
def Actividad3():
     return render_template('Actividad3.html')

@app.route('/Actividad4', methods=['POST'])
def Actividad4():

    temp_media = float(request.form['temp_media'])
    costo_pasaje = float(request.form['costo_pasaje'])

    input_data = np.array([[temp_media, costo_pasaje]])
    prediction = model.predict(input_data)

    return render_template('Actividad4.html', estimated_tourists=int(prediction[0]))

if __name__ == '__main__':
     app.run(debug=True)