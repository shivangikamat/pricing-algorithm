# ML-Based Safe Price Prediction System

## Overview

This system uses machine learning to determine optimal safe prices for CS2 skins based on historical data analysis and risk assessment.

## Key Features

### 1. **Dual-Model Approach**
   - **Classification Model (SVM)**: Predicts whether a skin is "risky" or "safe"
   - **Regression Model (Random Forest)**: Determines optimal safe price based on learned patterns

### 2. **Model Performance**
   - **SVM Risk Classification**: 98.66% accuracy
   - **Random Forest Safe Price Prediction**: 95.85% R² score (MSE: 126,162.72)

### 3. **Safety Constraints**

#### a. **3-Day Minimum Price Cap** ✅
   - **Critical constraint**: Safe price never exceeds the minimum price in the last 3 days
   - Prevents overpricing during recent market drops
   - Example:
     - Skin: "Sticker | Clan-Mystik (Holo) | Katowice 2014"
     - Mean price: $22,996
     - Formula safe price: $17,247 (25% discount)
     - 3-day minimum: $5,465
     - **ML safe price: $5,465** (capped at 3-day min)

#### b. **Price Bounds**
   - Minimum: 50% of market price
   - Maximum: 100% of market price
   - **With 3-day cap**: Safe price ≤ min(ML prediction, 3-day minimum)

### 4. **Data-Driven Features**

The system uses 5 key features:
1. `mean_price_usd`: Average historical price
2. `mean_pct_change`: Average daily percentage change
3. `volatility`: Standard deviation of price changes
4. `mean_deviation`: Deviation from rolling mean
5. `spike_count`: Number of price spikes detected

Plus the constraint feature:
6. `min_price_3day`: Minimum price in last 3 days

### 5. **Results**

#### Price Distribution
- Average market price: $241.24
- Average 3-day minimum: $194.96
- Average ML safe price: $174.33
- **Average discount: 30.02%**

#### Database
- Total skins analyzed: 26,835
- Risky skins detected: 3,748 (14.0%)
- Safe skins: 23,087 (86.0%)

## Implementation

### Data Pipeline

```
1. data_fetch.py
   ↓
   - Fetches historical price data from API
   - Saves raw daily prices
   - Calculates 3-day minimum prices
   
2. feature_engineering.ipynb
   ↓
   - Creates aggregated features per skin
   - Saves feature_df.csv
   
3. generate_database.py
   ↓
   - Trains SVM for risk classification
   - Trains Random Forest for safe price regression
   - Applies 3-day price cap
   - Saves complete database
```

### Flask Web Interface

The `app.py` automatically:
- Loads precomputed ML safe prices from database
- Applies 3-day cap for on-the-fly predictions
- Prioritizes ML-based prices over formula-based
- Falls back to formula-based pricing if models not available

### Files Generated

1. **data/feature_df.csv**: Aggregated features with 3-day minimums
2. **data/skin_database.csv**: Complete database with all predictions
3. **python_models/models/SVM_models/**:
   - `svm_best_model.pkl`: Risk classifier
   - `svm_scaler.pkl`: Feature scaler
   - `safe_price_regressor_RF.pkl`: Safe price regressor
4. **data/report/**:
   - `svm_safe_prices.csv`: Price comparison report
   - `price_comparison.csv`: Detailed analysis
   - `visualizations/`: Comparison charts

## Usage Example

```python
# Load database
import pandas as pd
df = pd.read_csv('data/skin_database.csv', index_col=0)

# Query a skin
skin_name = "AK-47 | Redline (Field-Tested)"
skin_data = df.loc[skin_name]

print(f"Market Price: ${skin_data['mean_price_usd']:.2f}")
print(f"3-Day Min: ${skin_data['min_price_3day']:.2f}")
print(f"ML Safe Price: ${skin_data['safe_price_usd_ML']:.2f}")
print(f"Risk: {'Risky' if skin_data['risk_score_SVM'] else 'Safe'}")
```

## Advantages of ML Approach

1. **Data-Driven**: Learns from 19,393+ skin profiles
2. **Context-Aware**: Considers multiple price characteristics
3. **Risk-Adjusted**: Combines volatility, spikes, and deviations
4. **Safe Bounds**: Respects 3-day minimum price constraint
5. **High Accuracy**: 95.85% R² on validation set

## Comparison: Formula vs ML

| Metric | Formula-Based | ML-Based |
|--------|--------------|----------|
| Average Discount | 25% (fixed) | 30.02% (learned) |
| Context Consideration | Limited | Full |
| Price Constraints | None | 3-day min enforced |
| Accuracy | N/A | 95.85% R² |

## Future Enhancements

1. Model retraining with new data
2. Real-time price updates
3. Seasonal price adjustments
4. Multi-provider price aggregation
5. A/B testing of safe price strategies

## Notes

- The 3-day minimum constraint is **critical** for safety
- ML prices often more conservative than fixed formula
- System favors risk mitigation over profit maximization
- All prices in USD based on multi-source aggregation

