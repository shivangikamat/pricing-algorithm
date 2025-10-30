#!/usr/bin/env python3
"""
Compare safe prices from all ML models
Combines predictions from BGMM, Logistic Regression, LDA, SVM, and Random Forest
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("COMPARING ALL ML MODELS")
print("=" * 80)

# 1. Load all model reports
reports_dir = "../data/report"
models_data = {}

# BGMM
try:
    bgmm_df = pd.read_csv(f"{reports_dir}/bgmm_safe_prices.csv", index_col=0)
    models_data['BGMM'] = bgmm_df[['safe_price_usd_BGMM', 'risk_score_BGMM']].rename(
        columns={'safe_price_usd_BGMM': 'safe_price', 'risk_score_BGMM': 'risk_score'}
    )
    print(f"✅ Loaded BGMM data: {len(bgmm_df)} skins")
except Exception as e:
    print(f"⚠️  BGMM not found: {e}")

# Logistic Regression
try:
    logreg_df = pd.read_csv(f"{reports_dir}/logistic_safe_prices.csv", index_col=0)
    models_data['Logistic Regression'] = logreg_df[['safe_price_usd_LogReg', 'risk_score_LogReg']].rename(
        columns={'safe_price_usd_LogReg': 'safe_price', 'risk_score_LogReg': 'risk_score'}
    )
    print(f"✅ Loaded Logistic Regression data: {len(logreg_df)} skins")
except Exception as e:
    print(f"⚠️  Logistic Regression not found: {e}")

# LDA
try:
    lda_df = pd.read_csv(f"{reports_dir}/lda_safe_prices.csv", index_col=0)
    # LDA uses different column names
    models_data['LDA'] = lda_df[['safe_price_usd', 'risk_score_model']].rename(
        columns={'safe_price_usd': 'safe_price', 'risk_score_model': 'risk_score'}
    )
    print(f"✅ Loaded LDA data: {len(lda_df)} skins")
except Exception as e:
    print(f"⚠️  LDA not found: {e}")

# SVM
try:
    svm_df = pd.read_csv(f"{reports_dir}/svm_safe_prices.csv", index_col=0)
    models_data['SVM'] = svm_df[['safe_price_usd_SVM', 'risk_score_SVM']].rename(
        columns={'safe_price_usd_SVM': 'safe_price', 'risk_score_SVM': 'risk_score'}
    )
    print(f"✅ Loaded SVM data: {len(svm_df)} skins")
except Exception as e:
    print(f"⚠️  SVM not found: {e}")

# Get mean_price_usd from one of the dataframes (they all have it)
mean_prices = None
for df_name, df in models_data.items():
    if 'mean_price_usd' in df.index:
        mean_prices = df.index
        break

if mean_prices is None:
    # Try to get from one of the actual dfs
    if len(bgmm_df) > 0:
        mean_prices = bgmm_df['mean_price_usd']

print()

# 2. Create unified comparison dataframe
print("📊 Creating unified comparison dataframe...")

# Get all common skin names across models
if models_data:
    common_skins = set(models_data[list(models_data.keys())[0]].index)
    for model_name in models_data.keys():
        common_skins = common_skins.intersection(set(models_data[model_name].index))
    
    print(f"   Found {len(common_skins)} skins common across all models")

# Build comparison dataframe
comparison_df = pd.DataFrame(index=list(common_skins))

# Add mean price from any of the original dataframes
if len(bgmm_df) > 0 and len(common_skins) > 0:
    comparison_df['mean_price_usd'] = bgmm_df.loc[list(common_skins), 'mean_price_usd']

# Add each model's predictions
for model_name, model_df in models_data.items():
    if len(common_skins) > 0:
        comparison_df[f'safe_price_{model_name}'] = model_df.loc[list(common_skins), 'safe_price']
        comparison_df[f'risk_score_{model_name}'] = model_df.loc[list(common_skins), 'risk_score']

# Add RF/ML model from skin_database if available
try:
    db_df = pd.read_csv("../data/skin_database.csv", index_col=0)
    if 'safe_price_usd_ML' in db_df.columns:
        common_with_db = comparison_df.index.intersection(db_df.index)
        comparison_df.loc[common_with_db, 'safe_price_Random Forest'] = db_df.loc[common_with_db, 'safe_price_usd_ML']
        print(f"✅ Added Random Forest (ML) predictions")
except Exception as e:
    print(f"⚠️  Random Forest not found: {e}")

print()
print(f"📊 Unified comparison ready: {len(comparison_df)} skins")
print()

# 3. Calculate statistics
print("=" * 80)
print("MODEL COMPARISON STATISTICS")
print("=" * 80)
print()

safe_price_cols = [col for col in comparison_df.columns if col.startswith('safe_price_')]
for col in safe_price_cols:
    model_name = col.replace('safe_price_', '')
    if model_name in comparison_df.columns.values and len(comparison_df[col].dropna()) > 0:
        discounts = (1 - comparison_df[col] / comparison_df['mean_price_usd']) * 100
        print(f"{model_name}:")
        print(f"   Avg discount: {discounts.mean():.2f}%")
        print(f"   Avg safe price: ${comparison_df[col].mean():.2f}")
        print()

# 4. Save unified comparison
output_path = f"{reports_dir}/all_models_comparison.csv"
comparison_df.to_csv(output_path, index=True)
print(f"💾 Saved unified comparison → {output_path}")

# 5. Create visualization for top 10 skins
print()
print("📊 Creating comparison visualization...")

# Select 10 interesting skins (mix of risky and safe, different price ranges)
if len(comparison_df) > 10:
    # Get a diverse sample
    sample_skins = comparison_df.nlargest(5, 'mean_price_usd').index.tolist()
    sample_skins += comparison_df.nsmallest(3, 'mean_price_usd').index.tolist()
    sample_skins += comparison_df.sample(2).index.tolist()
    sample_skins = list(set(sample_skins))[:10]
else:
    sample_skins = comparison_df.index.tolist()

comparison_sample = comparison_df.loc[sample_skins]

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Safe prices comparison
ax1 = axes[0]
x = np.arange(len(sample_skins))
width = 0.15

for i, col in enumerate(safe_price_cols):
    model_name = col.replace('safe_price_', '')
    ax1.bar(x + i*width, comparison_sample[col].values, width, label=model_name, alpha=0.8)

ax1.axhline(y=comparison_sample['mean_price_usd'].mean(), color='red', linestyle='--', 
           label='Avg Market Price', alpha=0.5)
ax1.set_xlabel('Skins', fontsize=11)
ax1.set_ylabel('Price (USD)', fontsize=11)
ax1.set_title('Safe Price Comparison Across Models (10 Sample Skins)', fontsize=12, fontweight='bold')
ax1.set_xticks(x + width * (len(safe_price_cols)-1) / 2)
ax1.set_xticklabels([name[:40] + '...' if len(name) > 40 else name for name in sample_skins], 
                    rotation=45, ha='right')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Discount comparison
ax2 = axes[1]
for col in safe_price_cols:
    model_name = col.replace('safe_price_', '')
    discounts = (1 - comparison_sample[col] / comparison_sample['mean_price_usd']) * 100
    ax2.plot(range(len(sample_skins)), discounts, marker='o', label=model_name, linewidth=2, markersize=6)

ax2.set_xlabel('Skins', fontsize=11)
ax2.set_ylabel('Discount (%)', fontsize=11)
ax2.set_title('Discount Comparison Across Models', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(sample_skins)))
ax2.set_xticklabels([name[:40] + '...' if len(name) > 40 else name for name in sample_skins], 
                    rotation=45, ha='right')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save visualization
os.makedirs(f"{reports_dir}/visualizations", exist_ok=True)
viz_path = f"{reports_dir}/visualizations/all_models_comparison.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"💾 Saved visualization → {viz_path}")

plt.show()

# 6. Create sample comparison table
print()
print("=" * 80)
print("SAMPLE COMPARISON (10 Skins)")
print("=" * 80)
print()

comparison_table = pd.DataFrame()
comparison_table['Mean Price'] = comparison_sample['mean_price_usd'].round(2)

for col in safe_price_cols:
    model_name = col.replace('safe_price_', '')
    comparison_table[f'{model_name} Safe Price'] = comparison_sample[col].round(2)
    discounts = (1 - comparison_sample[col] / comparison_sample['mean_price_usd']) * 100
    comparison_table[f'{model_name} Discount %'] = discounts.round(2)

comparison_table.index = [name[:60] + '...' if len(name) > 60 else name for name in comparison_table.index]
print(comparison_table.to_string())
print()

print("=" * 80)
print("COMPARISON COMPLETE! 🎉")
print("=" * 80)

