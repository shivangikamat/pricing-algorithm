from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "python_models", "models")

# Load trained models
# lda_model = joblib.load(os.path.join(MODEL_DIR, "LDA_model.pkl"))
# lda_scaler = joblib.load(os.path.join(MODEL_DIR, "LDA_scaler.pkl"))

# rf_model = joblib.load(os.path.join(MODEL_DIR, "RF_models", "random_forest.pkl"))
# svm_model = joblib.load(os.path.join(MODEL_DIR, "SVM_models", "SVM_best_model.pkl"))
# bgmm_model = joblib.load(os.path.join(MODEL_DIR, "BGMM_model", "bgmm_model.pkl"))
# bgmm_scaler = joblib.load(os.path.join(MODEL_DIR, "BGMM_model", "scaler.pkl"))

svm_model = joblib.load(os.path.join(MODEL_DIR, "SVM_models", "svm_best_model.pkl"))
svm_scaler = joblib.load(os.path.join(MODEL_DIR, "SVM_models", "svm_scaler.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_safe_price():
    # Get uploaded file
    file = request.files['file']
    if not file:
        return "Please upload a CSV file", 400
    
    df = pd.read_csv(file)
    
    # Validate feature columns
    required_cols = ['mean_price_usd', 'mean_pct_change', 'volatility', 'mean_deviation', 'spike_count']
    if not all(col in df.columns for col in required_cols):
        return f"Missing columns. Expected: {required_cols}", 400

    X = df[['mean_price_usd', 'mean_pct_change', 'volatility', 'mean_deviation', 'spike_count']]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = np.clip(X, -1e6, 1e6)

    # Scale using SVM scaler
    X_scaled = svm_scaler.transform(X)

    # Predict risk using SVM
    risk_pred_svm = svm_model.predict(X_scaled)
    alpha = 0.25

    # Safe price calculation
    safe_price = df["mean_price_usd"].values * (1 - alpha * risk_pred_svm)

    df["safe_price_usd"] = safe_price
    df["risk_score_svm"] = risk_pred_svm
    df["safe_discount_%"] = 100 * (1 - safe_price / df["mean_price_usd"])

    report_path = os.path.join(BASE_DIR, "..", "data", "report", "web_safe_price.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)

    return render_template('results.html', tables=[df.head(30).to_html(classes='table table-striped table-bordered', index=False)])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
