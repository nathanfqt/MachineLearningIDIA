from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
from werkzeug.utils import secure_filename
from ml_predict import Predictor 
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.joblib"
predictor = Predictor(MODEL_PATH)

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/webapp')
def webapp():
    """Page donnees"""
    return render_template('webapp.html')

@app.route('/result')
def result():
    """Page results"""
    return render_template('result.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template("result.html", prediction="No file uploaded")
    try:
        df = pd.read_csv(f)
        prediction = predictor.predict_from_dataframe(df)
        return render_template("result.html", prediction=prediction, error=None)
    except Exception as e:
        return render_template(
            "result.html",
            prediction=None,
            error="You didn't respect the template format. Please download the CSV template and try again."
        )

if __name__ == '__main__':
    app.run(debug=True)
