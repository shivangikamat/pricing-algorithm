#!/usr/bin/env python3
"""
Generate enhanced skin database with SVM predictions
This script runs the same logic as SVM_Model.ipynb to create the database
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

print("=" * 80)
print("GENERATING ENHANCED SKIN DATABASE WITH SVM PREDICTIONS")
print("=" * 80)

# 1️⃣ Load feature_df from CSV
feature_df_path = "../data/feature_df.csv"
feature_df = pd.read_csv(feature_df_path, index_col=0)
print(f"✅ Loaded feature_df from {feature_df_path}")
print(f"   Shape: {feature_df.shape}")
print(f"   Features: {list(feature_df.columns)}")
print()

# 2️⃣ Ensure risky_label exists
if "risky_label" not in feature_df.columns:
    spike_threshold = feature_df["spike_count"].quantile(0.9)
    vol_threshold   = feature_df["volatility"].quantile(0.9)
    feature_df["risky_label"] = np.where(
        (feature_df["spike_count"] >= spike_threshold) |
        (feature_df["volatility"]   >= vol_threshold),
        1, 0
    )

# 3️⃣ Prepare feature matrix and target
X = feature_df[[
    "mean_price_usd",
    "mean_pct_change",
    "volatility",
    "mean_deviation",
    "spike_count"
]].copy()
y = feature_df["risky_label"].values

# Clean
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = np.clip(X, -1e6, 1e6)
X = np.array(X, dtype=float)

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4️⃣ Train Fast SVM
print("🚀 Training Fast SVM (kernel='rbf', C=10, gamma='scale') ...")
svm = SVC(C=10, kernel="rbf", gamma="scale")
svm.fit(x_train_scaled, y_train)
print("✅ Training complete!")
print()

# 5️⃣ Evaluate
y_pred = svm.predict(x_test_scaled)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()

# 6️⃣ Save Model
os.makedirs("../python_models/models/SVM_models", exist_ok=True)
joblib.dump(svm, "../python_models/models/SVM_models/svm_best_model.pkl")
joblib.dump(scaler, "../python_models/models/SVM_models/svm_scaler.pkl")
print("💾 Saved model → ../python_models/models/SVM_models/svm_best_model.pkl")

# 7️⃣ Predict Risk using SVM
X_scaled = scaler.transform(X)
risk_pred = svm.predict(X_scaled)
feature_df["risk_score_SVM"] = risk_pred

# Calculate formula-based safe price as baseline
alpha = 0.25
feature_df["safe_price_usd_SVM"] = feature_df["mean_price_usd"] * (1 - alpha * feature_df["risk_score_SVM"])

# 8️⃣ Train ML model for optimal safe price prediction
print("\n" + "=" * 80)
print("TRAINING MACHINE LEARNING MODEL FOR OPTIMAL SAFE PRICE")
print("=" * 80)

# Create target: data-driven safe price based on risk and volatility
# Higher volatility and risk = larger discount from mean price
risk_weight = 0.25  # Base discount for risky skins
volatility_weight = 0.15  # Additional discount for high volatility

# Fill NaN values in volatility for safe price calculation
volatility_filled = feature_df["volatility"].fillna(0).values

# Create synthetic safe price targets using data-driven approach
safe_price_target = feature_df["mean_price_usd"].values * (
    1 - risk_pred * risk_weight - 
    volatility_filled * volatility_weight
)

# Ensure safe price is never negative and always < mean price
safe_price_target = np.clip(safe_price_target, 
                             feature_df["mean_price_usd"].values * 0.5,  # At least 50% of market
                             feature_df["mean_price_usd"].values * 1.0)  # Never above market

# Remove any NaN values and zero-priced skins before training
valid_mask = ~np.isnan(safe_price_target) & (feature_df["mean_price_usd"].values > 0)
print(f"   Valid samples for regression: {valid_mask.sum()} / {len(X_scaled)}")

# Filter to valid data
X_valid = X_scaled[valid_mask]
y_valid = safe_price_target[valid_mask]

# Train regression models to learn optimal safe price
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_valid, y_valid, test_size=0.2, random_state=42
)

print("\n🧠 Training Random Forest Regressor for safe price prediction...")
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_regressor.fit(X_reg_train, y_reg_train)
rf_pred = rf_regressor.predict(X_reg_test)
rf_mse = mean_squared_error(y_reg_test, rf_pred)
rf_r2 = r2_score(y_reg_test, rf_pred)
print(f"   ✅ Random Forest MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

print("\n🧠 Training Support Vector Regressor for safe price prediction...")
svr_regressor = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr_regressor.fit(X_reg_train, y_reg_train)
svr_pred = svr_regressor.predict(X_reg_test)
svr_mse = mean_squared_error(y_reg_test, svr_pred)
svr_r2 = r2_score(y_reg_test, svr_pred)
print(f"   ✅ SVR MSE: {svr_mse:.4f}, R²: {svr_r2:.4f}")

# Choose best model based on performance
if rf_r2 >= svr_r2:
    print("\n✅ Random Forest selected as best model!")
    best_regressor = rf_regressor
    best_name = "RF"
else:
    print("\n✅ SVR selected as best model!")
    best_regressor = svr_regressor
    best_name = "SVR"

# Predict safe prices using best model
safe_price_ml = best_regressor.predict(X_scaled)

# Ensure bounds: at least 50% of market price, at most 100%
safe_price_ml = np.clip(safe_price_ml,
                        feature_df["mean_price_usd"].values * 0.5,
                        feature_df["mean_price_usd"].values * 1.0)

# CRITICAL: Safe price should never exceed the minimum price in last 3 days
if 'min_price_3day' in feature_df.columns:
    safe_price_ml = np.minimum(safe_price_ml, feature_df["min_price_3day"].values)
    print(f"\n🔒 Applied 3-day price cap constraint")

feature_df["safe_price_usd_ML"] = safe_price_ml

print(f"\n📊 Safe Price Statistics:")
print(f"   Market price range: ${feature_df['mean_price_usd'].min():.2f} - ${feature_df['mean_price_usd'].max():.2f}")
print(f"   Safe price range: ${feature_df['safe_price_usd_ML'].min():.2f} - ${feature_df['safe_price_usd_ML'].max():.2f}")
print(f"   Average discount: {(1 - feature_df['safe_price_usd_ML'] / feature_df['mean_price_usd']).mean() * 100:.2f}%")

# Save regression model
model_dir = "../python_models/models/SVM_models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_regressor, os.path.join(model_dir, f"safe_price_regressor_{best_name}.pkl"))
print(f"💾 Saved best regression model → safe_price_regressor_{best_name}.pkl")
print("=" * 80)

# 9️⃣ Save Report
os.makedirs("../data/report", exist_ok=True)
svm_report = feature_df[[
    "mean_price_usd", "volatility", "spike_count",
    "risk_score_SVM", "safe_price_usd_SVM", "safe_price_usd_ML"
]].sort_values(by="risk_score_SVM", ascending=False)

svm_report.to_csv("../data/report/svm_safe_prices.csv", index=True)
print("\n💾 Saved → ../data/report/svm_safe_prices.csv")

# Save detailed comparison report
comparison_report = feature_df[[
    "mean_price_usd", "volatility", "spike_count",
    "risk_score_SVM"
]].copy()
comparison_report["formula_price"] = feature_df["safe_price_usd_SVM"]
comparison_report["ml_price"] = feature_df["safe_price_usd_ML"]
comparison_report["price_difference"] = feature_df["safe_price_usd_ML"] - feature_df["safe_price_usd_SVM"]
comparison_report = comparison_report.sort_values(by="risk_score_SVM", ascending=False)
comparison_report.to_csv("../data/report/price_comparison.csv", index=True)
print("💾 Saved → ../data/report/price_comparison.csv")
print()

# 🔟 Save Complete Database with Predictions
skin_database = feature_df.copy()
skin_database.index.name = 'skin_name'

database_path = "../data/skin_database.csv"
skin_database.to_csv(database_path, index=True)
print("✅ Saved complete skin database → skin_database.csv")
print(f"   Total skins: {len(skin_database)}")
print(f"   Risky skins: {feature_df['risk_score_SVM'].sum()}")
print(f"   Safe skins: {(feature_df['risk_score_SVM'] == 0).sum()}")
print()
print("📊 Database includes:")
print("   - All original features (mean_price_usd, volatility, etc.)")
print("   - SVM risk predictions (risk_score_SVM)")
print("   - Formula-based safe prices (safe_price_usd_SVM)")
print("   - ML-based safe prices (safe_price_usd_ML) ✨")
print("   - Ready for query interface!")
print()

print("=" * 80)
print("DATABASE GENERATION COMPLETE! 🎉")
print("=" * 80)

