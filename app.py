from flask import Flask
from flask import render_template

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


@app.route('/Caso1')
def Caso1():
     return render_template('Caso1.html')

@app.route('/Caso2')
def Caso2():
     return render_template('Caso2.html')

@app.route('/Caso3')
def Caso3():
     return render_template('Caso3.html')

@app.route('/Caso4')
def Caso4():
     return render_template('Caso4.html')
     

if __name__ == '__main__':
     app.run(debug=True)