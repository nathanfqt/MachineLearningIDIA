# Cyber Attack Classification Machine Learning Web App (GROUP 3 IDIA) 

This project presents a complete pipeline for cyber attack classification using Machine Learning, along with a Flask web application for inference.

The goal is to predict the type of cyber attack (**Intrusion**, **Malware**, or **DDoS**) from structured input data and provide a simple web interface to upload an CSV file and display the prediction.

---

## Features

- Machine Learning model for multi-class cyber attack classification
- Feature engineering and exploratory analysis in Jupyter notebooks
- Flask web application for model inference
- Drag & drop CSV upload interface
- Prediction result displayed on a dedicated results page

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/nathanfqt/MachineLearningIDIA.git
cd MachineLearningIDIA
```

2. Install the requirements
```bash
pip install -r requirements.txt
```

3. Run the application :)
```bash
python webapp/app.py
```

## Usage

- Open the web interface
- Drag & drop a CSV file into the upload area
- Submit the file
- The application displays the predicted attack type


## Machine Learning Workflow

The notebooks contain:
- Exploratory Data Analysis 
- Feature engineering
- Model training

## Technologies Used

- Python
- Flask
- Pandas / NumPy
- HTML / CSS / JavaScript
- Jupyter Notebook
