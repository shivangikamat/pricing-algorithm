# Model Comparison Guide

## Overview

This project now includes a comprehensive model comparison system that allows you to compare safe price predictions across multiple machine learning models.

## Available Models

1. **BGMM** (Bayesian Gaussian Mixture Model) - Unsupervised anomaly detection
2. **Logistic Regression** - Classification with risk probabilities
3. **LDA** (Linear Discriminant Analysis) - Dimensionality reduction + classification
4. **SVM** (Support Vector Machine) - Binary risk classification
5. **Random Forest** - Regression-based safe price prediction (ML optimized)

## How It Works

### 1. Generate Model Comparisons

Run the comparison script:

```bash
cd notebooks
python compare_all_models.py
```

This will:
- Load predictions from all trained models
- Create a unified comparison database
- Generate visualization charts
- Save results to `data/report/all_models_comparison.csv`

### 2. Access via Web Interface

Navigate to the query page and search for any skin. The results will now include:
- Primary model prediction (SVM with 3-day cap)
- **New**: Comparison table with all models' predictions

### 3. API Endpoints

#### Get Single Model Prediction
```bash
POST /api/get_skin_stats
{
  "skin_name": "AK-47 | Redline (Field-Tested)"
}
```

#### Get All Models Comparison
```bash
POST /api/get_all_models
{
  "skin_name": "AK-47 | Redline (Field-Tested)"
}
```

Response:
```json
{
  "skin_name": "AK-47 | Redline (Field-Tested)",
  "mean_price_usd": 35.31,
  "models": {
    "BGMM": {
      "safe_price_usd": 33.45,
      "discount_pct": 5.27
    },
    "Logistic Regression": {
      "safe_price_usd": 33.12,
      "discount_pct": 6.20
    },
    "LDA": {
      "safe_price_usd": 34.89,
      "discount_pct": 1.19
    },
    "SVM": {
      "safe_price_usd": 35.31,
      "discount_pct": 0.00
    },
    "Random Forest": {
      "safe_price_usd": 28.00,
      "discount_pct": 20.71
    }
  }
}
```

## Key Differences Between Models

### BGMM (Bayesian Gaussian Mixture Model)
- **Approach**: Unsupervised learning, anomaly detection
- **Risk Scoring**: Continuous score (0-1) based on likelihood
- **Discount Range**: Variable, depends on anomaly score
- **Pros**: No labels needed, finds hidden patterns
- **Cons**: Less interpretable, sensitive to outliers

### Logistic Regression
- **Approach**: Supervised classification with probabilities
- **Risk Scoring**: Probability of being risky (0-1)
- **Discount Range**: 25% for risky, 0% for safe
- **Pros**: Interpretable, fast, good baseline
- **Cons**: Linear decision boundary

### LDA (Linear Discriminant Analysis)
- **Approach**: Supervised dimensionality reduction + classification
- **Risk Scoring**: Classification (0 or 1)
- **Discount Range**: Variable based on classification
- **Pros**: Reduces dimensions, good for high-dimensional data
- **Cons**: Assumes Gaussian distribution

### SVM (Support Vector Machine)
- **Approach**: Binary classification with non-linear kernel
- **Risk Scoring**: Binary classification (0 or 1)
- **Discount Range**: 25% for risky, 0% for safe (formula-based)
- **Pros**: High accuracy (98.66%), handles non-linear patterns
- **Cons**: Binary only, no probability scores

### Random Forest (ML Optimized)
- **Approach**: Ensemble regression trained on 19,393 skins
- **Risk Scoring**: Continuous prediction (0-1) via regression
- **Discount Range**: Learned from data, avg 30.02%
- **Pros**: Highest accuracy (95.85% R²), data-driven, 3-day cap applied
- **Cons**: Slower, more complex

## Model Statistics

Based on 26,706 skins common across all models:

| Model | Avg Discount | Avg Safe Price | Approach |
|-------|--------------|----------------|----------|
| BGMM | Variable | Based on anomaly | Unsupervised |
| Logistic Regression | 0-25% | Probability-based | Supervised |
| LDA | Variable | Classification-based | Dimensionality reduction |
| SVM | 0-25% | Binary classification | Supervised, non-linear |
| Random Forest | 30.02% | Regression (ML) | Ensemble, data-driven |

## Visualization

The comparison script generates:
- Bar charts comparing safe prices across models for 10 sample skins
- Line plots showing discount percentages
- Saved to `data/report/visualizations/all_models_comparison.png`

## Integration

### In Flask App

The web interface automatically:
1. Loads `all_models_comparison.csv` on startup
2. Shows primary model (SVM + RF) prediction first
3. Displays comparison table below with all models
4. Updates live when new comparisons are generated

### In Your Code

```python
import pandas as pd

# Load comparison data
comparison_df = pd.read_csv('data/report/all_models_comparison.csv', index_col=0)

# Get predictions for a skin
skin_name = "AK-47 | Redline (Field-Tested)"
skin_predictions = comparison_df.loc[skin_name]

# Access individual model predictions
bgmm_price = skin_predictions['safe_price_BGMM']
rf_price = skin_predictions['safe_price_Random Forest']
```

## Rerunning Comparisons

After updating any model:

```bash
# 1. Regenerate individual model reports (run model notebooks)
# 2. Generate unified comparison
cd notebooks
python compare_all_models.py

# 3. Restart Flask app
cd ../src
python app.py
```

## Notes

- Random Forest includes the 3-day minimum price cap
- SVM is used for binary risk classification
- BGMM is unsupervised and finds anomalies without labels
- All models use the same feature set for consistency
- Safe prices are capped at 50% of market minimum

## Future Enhancements

- Voting ensemble across models
- Weighted average based on model accuracy
- Model confidence scores
- Historical accuracy tracking
- Real-time model retraining

