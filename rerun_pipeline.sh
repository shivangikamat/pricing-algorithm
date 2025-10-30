#!/bin/bash

###############################################################################
# CS2 Skin Pricing Pipeline - Rerun Script
# 
# This script reruns the entire pipeline after changes to data_fetch.py or 
# other data processing components.
###############################################################################

set -e  # Exit on error


echo "🔄 Starting CS2 Skin Pricing Pipeline Rerun"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NOTEBOOKS_DIR="$SCRIPT_DIR/notebooks"

# Check if we're in the right directory
if [ ! -f "$NOTEBOOKS_DIR/data_fetch.py" ]; then
    echo "❌ Error: Could not find notebooks/data_fetch.py"
    echo "   Please run this script from the pricing-algorithm root directory"
    exit 1
fi

# Step 1: Data Fetching
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Running data_fetch.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$NOTEBOOKS_DIR"
python data_fetch.py

if [ $? -ne 0 ]; then
    echo "❌ Error: data_fetch.py failed"
    exit 1
fi

echo ""
echo "✅ Step 1 complete: Features extracted"
echo ""

# Step 2: Generate Database with ML Models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Generating database with ML models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python generate_database.py

if [ $? -ne 0 ]; then
    echo "❌ Error: generate_database.py failed"
    exit 1
fi

echo ""
echo "✅ Step 2 complete: Database and models generated"
echo ""

# Step 3: Verification
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Verifying outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$SCRIPT_DIR"

# Check for key files
echo -n "Checking data/feature_df.csv... "
if [ -f "data/feature_df.csv" ]; then
    echo "✓"
else
    echo "⚠ Missing"
fi

echo -n "Checking data/skin_database.csv... "
if [ -f "data/skin_database.csv" ]; then
    echo "✓"
else
    echo "⚠ Missing"
fi

echo -n "Checking models... "
if [ -f "python_models/models/SVM_models/svm_best_model.pkl" ] && \
   [ -f "python_models/models/SVM_models/safe_price_regressor_RF.pkl" ]; then
    echo "✓"
else
    echo "⚠ Missing"
fi


echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PIPELINE RERUN COMPLETE! ✅"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Next steps:"
echo "   1. Restart your Flask app: cd src && python app.py"
echo "   2. Or test the API: curl -X POST http://localhost:5000/api/get_skin_stats -H 'Content-Type: application/json' -d '{\"skin_name\": \"AK-47\"}'"
echo ""

