from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "python_models", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Load trained models
svm_model = joblib.load(os.path.join(MODEL_DIR, "SVM_models", "svm_best_model.pkl"))
svm_scaler = joblib.load(os.path.join(MODEL_DIR, "SVM_models", "svm_scaler.pkl"))

# Try to load ML-based safe price regression model
regressor_path_rf = os.path.join(MODEL_DIR, "SVM_models", "safe_price_regressor_RF.pkl")
regressor_path_svr = os.path.join(MODEL_DIR, "SVM_models", "safe_price_regressor_SVR.pkl")

if os.path.exists(regressor_path_rf):
    safe_price_regressor = joblib.load(regressor_path_rf)
    print("✅ Loaded ML-based safe price regressor (Random Forest)")
elif os.path.exists(regressor_path_svr):
    safe_price_regressor = joblib.load(regressor_path_svr)
    print("✅ Loaded ML-based safe price regressor (SVR)")
else:
    safe_price_regressor = None
    print("⚠️  No ML regressor found, will use formula-based prices")

# Load feature database (try skin_database.csv first, fallback to feature_df.csv)
DATABASE_PATH = os.path.join(DATA_DIR, "skin_database.csv")
FEATURE_DF_PATH = os.path.join(DATA_DIR, "feature_df.csv")

if os.path.exists(DATABASE_PATH):
    feature_df = pd.read_csv(DATABASE_PATH, index_col=0)
    print(f"✅ Loaded enhanced database with predictions from {DATABASE_PATH}")
else:
    feature_df = pd.read_csv(FEATURE_DF_PATH, index_col=0)
    print(f"⚠️  Using basic feature database from {FEATURE_DF_PATH}")
    print("   Run SVM_Model.ipynb to generate enhanced database with predictions")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query')
def query_skin():
    return render_template('query.html')

@app.route('/api/search_skins', methods=['POST'])
def search_skins():
    """API endpoint to search for skins by name"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    # If no query provided, return random skins
    if not query:
        # Return first 100 skins for random selection
        results = feature_df.head(100).index.tolist()
    else:
        # Search for matching skins (case-insensitive, partial match)
        matching_skins = feature_df[feature_df.index.str.contains(query, case=False, na=False)]
        # Return up to 50 results
        results = matching_skins.head(50).index.tolist()
    
    return jsonify({'results': results})

@app.route('/api/get_skin_stats', methods=['POST'])
def get_skin_stats():
    """API endpoint to get statistics and predictions for a specific skin"""
    data = request.get_json()
    skin_name = data.get('skin_name', '').strip()
    
    if not skin_name:
        return jsonify({'error': 'Skin name is required'}), 400
    
    # Check if skin exists
    if skin_name not in feature_df.index:
        return jsonify({'error': 'Skin not found in database'}), 404
    
    # Get skin features
    skin_features = feature_df.loc[skin_name]
    
    # Check if we have precomputed predictions in database
    if 'risk_score_SVM' in feature_df.columns:
        # Use precomputed risk score
        risk_pred = int(skin_features['risk_score_SVM'])
        
        # Prefer ML-based safe price if available, otherwise use formula-based
        if 'safe_price_usd_ML' in feature_df.columns:
            safe_price = round(skin_features['safe_price_usd_ML'], 2)
        elif 'safe_price_usd_SVM' in feature_df.columns:
            safe_price = round(skin_features['safe_price_usd_SVM'], 2)
        else:
            # Calculate formula-based safe price
            alpha = 0.25
            mean_price = skin_features['mean_price_usd']
            safe_price = round(mean_price * (1 - alpha * risk_pred), 2)
    else:
        # Calculate predictions on the fly
        X = pd.DataFrame({
            'mean_price_usd': [skin_features['mean_price_usd']],
            'mean_pct_change': [skin_features['mean_pct_change']],
            'volatility': [skin_features['volatility']],
            'mean_deviation': [skin_features['mean_deviation']],
            'spike_count': [skin_features['spike_count']]
        })
        
        # Clean and scale
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = np.clip(X, -1e6, 1e6)
        X_scaled = svm_scaler.transform(X)
        
        # Predict risk
        risk_pred = int(svm_model.predict(X_scaled)[0])
        
        # Use ML regressor if available, otherwise fallback to formula
        if safe_price_regressor is not None:
            safe_price_ml = safe_price_regressor.predict(X_scaled)[0]
            mean_price = skin_features['mean_price_usd']
            safe_price = np.clip(safe_price_ml, mean_price * 0.5, mean_price * 1.0)
            
            # Apply 3-day price cap if available
            if 'min_price_3day' in skin_features.index:
                safe_price = min(safe_price, skin_features['min_price_3day'])
            
            safe_price = round(safe_price, 2)
        else:
            alpha = 0.25
            mean_price = skin_features['mean_price_usd']
            safe_price = mean_price * (1 - alpha * risk_pred)
            
            # Apply 3-day price cap if available
            if 'min_price_3day' in skin_features.index:
                safe_price = min(safe_price, skin_features['min_price_3day'])
            
            safe_price = round(safe_price, 2)
    
    # Calculate discount
    mean_price = skin_features['mean_price_usd']
    discount = round(100 * (1 - safe_price / mean_price), 2) if mean_price > 0 else 0
    
    # Prepare response
    stats = {
        'skin_name': skin_name,
        'mean_price_usd': round(mean_price, 2),
        'mean_pct_change': round(skin_features['mean_pct_change'], 4),
        'volatility': round(skin_features['volatility'], 4),
        'mean_deviation': round(skin_features['mean_deviation'], 4),
        'spike_count': int(skin_features['spike_count']),
        'risk_score': risk_pred,
        'risk_status': 'Risky' if risk_pred == 1 else 'Safe',
        'safe_price_usd': safe_price,
        'discount_pct': discount
    }
    
    return jsonify(stats)

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
