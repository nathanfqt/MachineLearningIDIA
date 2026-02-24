from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    """Page d'accueil - Overview du reservoir"""
    return render_template('index.html')

@app.route('/webapp')
def webapp():
    """Page donnees experimentales"""
    return render_template('webapp.html')

@app.route('/result')
def result():
    """Page results"""
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
