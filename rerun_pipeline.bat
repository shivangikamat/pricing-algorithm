@echo off
REM CS2 Skin Pricing Pipeline - Windows Batch Script

echo 🔄 Starting CS2 Skin Pricing Pipeline Rerun
echo.

REM Get the directory where the script is located
cd /d "%~dp0"

REM Check if we're in the right directory
if not exist "notebooks\data_fetch.py" (
    echo ❌ Error: Could not find notebooks\data_fetch.py
    echo    Please run this script from the pricing-algorithm root directory
    pause
    exit /b 1
)

REM Step 1: Data Fetching
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 1⃣  Running data_fetch.py
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cd notebooks
python data_fetch.py
if errorlevel 1 (
    echo ❌ Error: data_fetch.py failed
    pause
    exit /b 1
)

echo.
echo ✅ Step 1 complete: Features extracted
echo.

REM Step 2: Generate Database with ML Models
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 2⃣  Generating database with ML models
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python generate_database.py
if errorlevel 1 (
    echo ❌ Error: generate_database.py failed
    pause
    exit /b 1
)

echo.
echo ✅ Step 2 complete: Database and models generated
echo.

REM Step 3: Verification
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 3⃣  Verifying outputs
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cd ..

echo Checking data\feature_df.csv...
if exist "data\feature_df.csv" (
    echo    ✓ Found
) else (
    echo    ⚠ Missing
)

echo Checking data\skin_database.csv...
if exist "data\skin_database.csv" (
    echo    ✓ Found
) else (
    echo    ⚠ Missing
)

echo Checking models...
if exist "python_models\models\SVM_models\svm_best_model.pkl" (
    if exist "python_models\models\SVM_models\safe_price_regressor_RF.pkl" (
        echo    ✓ Found
    ) else (
        echo    ⚠ Missing
    )
) else (
    echo    ⚠ Missing
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ✅ PIPELINE RERUN COMPLETE! ✅
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 📝 Next steps:
echo    1. Restart your Flask app: cd src ^&^& python app.py
echo    2. Or test the API with PowerShell or curl
echo.

pause

