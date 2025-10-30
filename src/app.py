from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load trained models
lda_model = joblib.load(os.path.join(MODEL_DIR, "LDA_model.pkl"))
lda_scaler = joblib.load(os.path.join(MODEL_DIR, "LDA_scaler.pkl"))

rf_model = joblib.load(os.path.join(MODEL_DIR, "RF_models", "random_forest.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "SVM_models", "SVM_best_model.pkl"))
bgmm_model = joblib.load(os.path.join(MODEL_DIR, "BGMM_model", "bgmm_model.pkl"))
bgmm_scaler = joblib.load(os.path.join(MODEL_DIR, "BGMM_model", "scaler.pkl"))

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
    required_cols = ['mean_price_usd', 'mean_pct_change', 'volatility', 'mean_deviation', 'spike_count', 'recent_3day_mean']
    if not all(col in df.columns for col in required_cols):
        return f"Missing columns. Expected: {required_cols}", 400

    X = df[['mean_price_usd', 'mean_pct_change', 'volatility', 'mean_deviation', 'spike_count']]
    X = np.clip(X.fillna(0), -1e6, 1e6)

    # Scale using LDA scaler
    X_scaled = lda_scaler.transform(X)

    # Predict risk using LDA
    risk_prob_lda = lda_model.predict_proba(X_scaled)[:, 1]
    alpha = 0.25

    # Safe price logic with constraint
    safe_price = np.minimum(
        df["mean_price_usd"].values * (1 - alpha * risk_prob_lda),
        df["recent_3day_mean"].values
    )

    df["safe_price_usd"] = safe_price
    df["risk_score_lda"] = risk_prob_lda
    df["safe_discount_%"] = 100 * (1 - safe_price / df["mean_price_usd"])

    report_path = os.path.join(BASE_DIR, "../data/report/web_safe_price.csv")
    df.to_csv(report_path, index=False)

    return render_template('results.html', tables=[df.head(30).to_html(classes='table table-striped table-bordered', index=False)])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
