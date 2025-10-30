#!/usr/bin/env python3
"""
CS2 Skin Pricing Pipeline - Rerun Script (Cross-platform)
Works on Windows, macOS, and Linux
"""

import os
import sys
import subprocess

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔄 Starting CS2 Skin Pricing Pipeline Rerun")
    print("")
    
    # Get the directory where the script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    NOTEBOOKS_DIR = os.path.join(SCRIPT_DIR, "notebooks")
    
    # Check if we're in the right directory
    if not os.path.isfile(os.path.join(NOTEBOOKS_DIR, "data_fetch.py")):
        print("❌ Error: Could not find notebooks/data_fetch.py")
        print("   Please run this script from the pricing-algorithm root directory")
        sys.exit(1)
    
    # Step 1: Data Fetching
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("1️⃣  Running data_fetch.py")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if not run_command("python data_fetch.py", cwd=NOTEBOOKS_DIR):
        print("❌ Error: data_fetch.py failed")
        sys.exit(1)
    
    print("")
    print("✅ Step 1 complete: Features extracted")
    print("")
    
    # Step 2: Generate Database with ML Models
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("2️⃣  Generating database with ML models")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if not run_command("python generate_database.py", cwd=NOTEBOOKS_DIR):
        print("❌ Error: generate_database.py failed")
        sys.exit(1)
    
    print("")
    print("✅ Step 2 complete: Database and models generated")
    print("")
    
    # Step 3: Verification
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("3️⃣  Verifying outputs")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Check for key files
    print("Checking data/feature_df.csv... ", end="")
    if os.path.isfile(os.path.join(SCRIPT_DIR, "data", "feature_df.csv")):
        print("✓")
    else:
        print("⚠ Missing")
    
    print("Checking data/skin_database.csv... ", end="")
    if os.path.isfile(os.path.join(SCRIPT_DIR, "data", "skin_database.csv")):
        print("✓")
    else:
        print("⚠ Missing")
    
    print("Checking models... ", end="")
    if (os.path.isfile(os.path.join(SCRIPT_DIR, "python_models", "models", "SVM_models", "svm_best_model.pkl")) and
        os.path.isfile(os.path.join(SCRIPT_DIR, "python_models", "models", "SVM_models", "safe_price_regressor_RF.pkl"))):
        print("✓")
    else:
        print("⚠ Missing")
    
    # Show database stats
    print("")
    print("📊 Database Statistics:")
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "skin_database.csv"), index_col=0)
        risky = df['risk_score_SVM'].sum()
        safe = (df['risk_score_SVM'] == 0).sum()
        avg_discount = (1 - df['safe_price_usd_ML'] / df['mean_price_usd']).mean() * 100
        print(f"   Total skins: {len(df):,}")
        print(f"   Risky: {risky:,} ({risky/len(df)*100:.1f}%)")
        print(f"   Safe: {safe:,} ({safe/len(df)*100:.1f}%)")
        print(f"   Avg discount: {avg_discount:.2f}%")
    except Exception as e:
        print(f"   Could not load database: {e}")
    
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ PIPELINE RERUN COMPLETE! ✅")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    print("📝 Next steps:")
    print("   1. Restart your Flask app: cd src && python app.py")
    print("   2. Or test the API: curl -X POST http://localhost:5000/api/get_skin_stats -H 'Content-Type: application/json' -d '{\"skin_name\": \"AK-47\"}'")
    print("")

if __name__ == "__main__":
    main()

