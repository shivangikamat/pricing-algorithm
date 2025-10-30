#!/usr/bin/env python3
"""
Visualize comparison between formula-based and ML-based safe price predictions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("PRICE COMPARISON VISUALIZATION")
print("=" * 80)

# Load the enhanced database
database_path = "../data/skin_database.csv"
comparison_path = "../data/report/price_comparison.csv"

skin_db = pd.read_csv(database_path, index_col=0)
comparison_df = pd.read_csv(comparison_path, index_col=0)

print(f"✅ Loaded database: {len(skin_db)} skins")
print()

# Filter to risky skins for better visualization
risky_skins = comparison_df[comparison_df['risk_score_SVM'] == 1].copy()

print("📊 Price Comparison Statistics (Risky Skins Only):")
print(f"   Total risky skins: {len(risky_skins)}")
print(f"   Average formula discount: {(1 - risky_skins['formula_price'] / risky_skins['mean_price_usd']).mean() * 100:.2f}%")
print(f"   Average ML discount: {(1 - risky_skins['ml_price'] / risky_skins['mean_price_usd']).mean() * 100:.2f}%")
print(f"   Average price difference: ${risky_skins['price_difference'].abs().mean():.2f}")
print()

# Create visualization directory
os.makedirs("../data/report/visualizations", exist_ok=True)

# 1. Scatter plot: Formula vs ML prices
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(risky_skins['formula_price'], risky_skins['ml_price'], 
           alpha=0.5, s=30, c=risky_skins['volatility'], cmap='viridis')
plt.plot([0, risky_skins['formula_price'].max()], [0, risky_skins['formula_price'].max()], 
         'r--', alpha=0.5, label='Perfect Match')
plt.xlabel('Formula-Based Safe Price ($)', fontsize=11)
plt.ylabel('ML-Based Safe Price ($)', fontsize=11)
plt.title('Formula vs ML Safe Prices (Risky Skins)', fontsize=12, fontweight='bold')
plt.colorbar(label='Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Distribution of price differences
plt.subplot(2, 2, 2)
price_diff = risky_skins['price_difference']
plt.hist(price_diff, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
plt.xlabel('Price Difference (ML - Formula) ($)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Price Differences', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Discount comparison
plt.subplot(2, 2, 3)
formula_discount = (1 - risky_skins['formula_price'] / risky_skins['mean_price_usd']) * 100
ml_discount = (1 - risky_skins['ml_price'] / risky_skins['mean_price_usd']) * 100

plt.scatter(formula_discount, ml_discount, alpha=0.5, s=30, c='purple')
plt.plot([0, 50], [0, 50], 'r--', alpha=0.5, label='Perfect Match')
plt.xlabel('Formula Discount (%)', fontsize=11)
plt.ylabel('ML Discount (%)', fontsize=11)
plt.title('Discount Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Box plot: Discount by spike count
plt.subplot(2, 2, 4)
discount_df = pd.DataFrame({
    'Formula': formula_discount,
    'ML': ml_discount
})
discount_df.boxplot(ax=plt.gca())
plt.ylabel('Discount (%)', fontsize=11)
plt.title('Discount Distribution: Formula vs ML', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../data/report/visualizations/price_comparison.png", dpi=300, bbox_inches='tight')
print("💾 Saved → ../data/report/visualizations/price_comparison.png")

plt.show()

# 5. Analyze top differences
print("\n📈 Top 20 Skins with Largest Price Differences (ML vs Formula):")
print("=" * 80)
top_diff = comparison_df.nlargest(20, 'price_difference')[['mean_price_usd', 'formula_price', 'ml_price', 'price_difference', 'volatility']]
top_diff.index = top_diff.index.str[:50]  # Truncate long names
print(top_diff.to_string())
print()

# 6. Summary statistics
print("\n📊 Summary Statistics:")
print("=" * 80)
stats_summary = {
    'Metric': [
        'Mean Formula Price',
        'Mean ML Price', 
        'Mean Difference',
        'RMSE',
        'Mean Absolute Difference',
        'Formula > ML (%)',
        'ML > Formula (%)'
    ],
    'Value': [
        f"${comparison_df['formula_price'].mean():.2f}",
        f"${comparison_df['ml_price'].mean():.2f}",
        f"${comparison_df['price_difference'].mean():.2f}",
        f"${np.sqrt((comparison_df['price_difference']**2).mean()):.2f}",
        f"${comparison_df['price_difference'].abs().mean():.2f}",
        f"{(comparison_df['price_difference'] < 0).sum() / len(comparison_df) * 100:.1f}%",
        f"{(comparison_df['price_difference'] > 0).sum() / len(comparison_df) * 100:.1f}%"
    ]
}
stats_df = pd.DataFrame(stats_summary)
print(stats_df.to_string(index=False))
print()

print("=" * 80)
print("VISUALIZATION COMPLETE! 🎉")
print("=" * 80)

