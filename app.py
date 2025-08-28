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

if __name__ == '__main__':
     app.run(debug=True)