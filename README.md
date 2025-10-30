# CS2 Skin Pricing Algorithm

A machine learning-powered system for predicting safe prices of CS2 skins with risk assessment.

## Features

- 🤖 **ML-Based Safe Price Prediction**: Random Forest regression (95.85% R² accuracy)
- ⚠️ **Risk Classification**: SVM classifier (98.66% accuracy)
- 🔒 **3-Day Price Cap**: Safe prices never exceed last 3 days' minimum
- 📊 **26,835+ Skins**: Comprehensive database with historical analysis
- 🌐 **Web Interface**: Flask-based query system
- 🎯 **Multi-Model Comparison**: Compare 5 different ML models side-by-side
- 📈 **Performance Analytics**: Detailed model statistics and rankings

## Model Performance Summary

| Model | Avg Discount | Consistency | Best For |
|-------|--------------|-------------|----------|
| **Random Forest** ⭐ | 31.44% | 97.79% CV | Maximum protection |
| **BGMM** ⭐⭐⭐⭐⭐ | 0.93% | 97.06% CV | Balanced risk assessment |
| Logistic Regression | 4.91% | 164.41% CV | Baseline comparison |
| SVM | 4.68% | 1108.11% CV | Classification only |

**See `data/report/BEST_MODEL_ANALYSIS.md` for complete analysis**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file in the root directory:
```
API_KEY=your_price_empire_api_key_here
```

### 3. Run the Pipeline

**Option A: Automated Script (Cross-platform)**

- **Windows**: Double-click `rerun_pipeline.bat` or run in Command Prompt
- **macOS/Linux**: `./rerun_pipeline.sh` or `python rerun_pipeline.py`
- **Any OS**: `python rerun_pipeline.py` (works everywhere!)

**Option B: Manual Steps**
```bash
# Step 1: Fetch data and extract features
cd notebooks
python data_fetch.py

# Step 2: Generate database with ML models
python generate_database.py

# Step 3: Start Flask app
cd ../src
python app.py
```

### 4. Access Web Interface

Open your browser to: `http://localhost:5000`

## File Structure

```
pricing-algorithm/
├── data/
│   ├── feature_df.csv              # Aggregated skin features
│   ├── skin_database.csv           # Complete database with predictions
│   ├── raw/                        # Raw historical data (gitignored)
│   └── report/                     # Analysis reports
├── notebooks/
│   ├── data_fetch.py               # Data fetching and preprocessing
│   ├── generate_database.py        # ML model training
│   ├── SVM_Model.ipynb             # SVM classification
│   └── feature_engineering.ipynb   # Feature analysis
├── src/
│   ├── app.py                      # Flask web application
│   └── templates/                  # HTML templates
├── python_models/models/           # Trained ML models (gitignored)
├── rerun_pipeline.sh               # Automated pipeline script
└── PIPELINE_RERUN_GUIDE.md         # Detailed pipeline documentation
```

## Pipeline Overview

```
Data Fetching → Feature Engineering → ML Training → Web Interface
```

1. **data_fetch.py**: Fetches historical prices, calculates features, 3-day minimums
2. **generate_database.py**: Trains SVM + Random Forest, applies constraints
3. **app.py**: Flask web interface for querying skins

## Making Changes

### After Modifying `data_fetch.py`:

Simply rerun the pipeline:
```bash
./rerun_pipeline.sh
```

This will:
- ✅ Fetch new data
- ✅ Recalculate features
- ✅ Retrain ML models
- ✅ Regenerate database
- ✅ Show verification stats

### Restart Flask App:
```bash
cd src
python app.py
```

## Key Statistics

- **Total Skins**: 26,835
- **Risky Skins**: 3,748 (14.0%)
- **Safe Skins**: 23,087 (86.0%)
- **Average Discount**: 30.02%
- **ML Model R²**: 95.85%
- **SVM Accuracy**: 98.66%

## Important Constraints

### 3-Day Minimum Price Cap ⚠️

**Critical**: Safe prices never exceed the minimum price in the last 3 days. This prevents overpricing during recent market drops.

Example:
- Skin: "Sticker | Clan-Mystik (Holo) | Katowice 2014"
- Market price: $22,996
- Formula safe price: $17,247 (25% discount)
- **3-day minimum: $5,465**
- **ML safe price: $5,465** ✅ (capped)

## API Endpoints

### Query Single Skin
```bash
POST /api/get_skin_stats
Content-Type: application/json

{
  "skin_name": "AK-47 | Redline (Field-Tested)"
}
```

### Search Skins
```bash
POST /api/search_skins
Content-Type: application/json

{
  "query": "AK-47"
}
```

## Dependencies

- **Python 3.8+**
- Flask (web framework)
- scikit-learn (ML models)
- pandas, numpy (data processing)
- joblib (model persistence)
- requests (API calls)

## Documentation

- **PIPELINE_RERUN_GUIDE.md**: Detailed pipeline documentation
- **ML_SAFE_PRICE_SUMMARY.md**: ML approach explanation
- Inline code comments

## Notes

- Historical data file (~212MB) is gitignored
- Models are regenerated on each pipeline run
- Flask app automatically reloads database on startup
- API key required for `price-empire.com`

## Troubleshooting

### "API rate limit exceeded"
Wait a few minutes before rerunning the pipeline.

### "Model not found"
Run `generate_database.py` to regenerate models.

### "Git large file error"
The `price_history.csv` is gitignored. Don't commit it.

### "Flask shows old data"
Restart the Flask app after running the pipeline.

### Windows-specific issues

**"Permission denied" when running `.sh` file**
- Use `rerun_pipeline.bat` instead (Windows batch file)
- Or use `python rerun_pipeline.py` (cross-platform Python script)

**"python is not recognized"**
- Install Python 3.8+ from [python.org](https://www.python.org/)
- Or use `py rerun_pipeline.py` instead

**Path issues with backslashes**
- Use forward slashes `/` or escaped backslashes `\\` in paths
- The Python script handles this automatically

## Platform-Specific Notes

| Platform | Best Method | Script |
|----------|-------------|--------|
| Windows | `.bat` or Python | `rerun_pipeline.bat` or `rerun_pipeline.py` |
| macOS | Shell or Python | `rerun_pipeline.sh` or `rerun_pipeline.py` |
| Linux | Shell or Python | `rerun_pipeline.sh` or `rerun_pipeline.py` |
| **Any** | Python | `rerun_pipeline.py` ✅ |

## License

MIT

## Contributors

Created for CS2 skin pricing analysis and risk assessment.
