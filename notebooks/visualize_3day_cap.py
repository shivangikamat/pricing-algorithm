#!/usr/bin/env python3
"""
Visualize the impact of the 3-day price cap on safe prices
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 80)
print("3-DAY PRICE CAP ANALYSIS")
print("=" * 80)

# Load the enhanced database
database_path = "../data/skin_database.csv"
skin_db = pd.read_csv(database_path, index_col=0)

print(f"✅ Loaded database: {len(skin_db)} skins")
print()

# Analyze impact of 3-day cap
skin_db['safe_price_before_cap'] = skin_db['safe_price_usd_ML']
skin_db['cap_applied'] = skin_db['safe_price_usd_ML'] < skin_db['safe_price_before_cap'].where(
    lambda x: x > skin_db['min_price_3day'], 
    skin_db['safe_price_before_cap']
)

# Filter to skins where cap was applied
capped_skins = skin_db[skin_db['safe_price_usd_ML'] < skin_db.apply(
    lambda row: min(row['mean_price_usd'] * 0.5, row.get('safe_price_before_cap', row['mean_price_usd'])), axis=1
)]

print("📊 3-Day Cap Impact:")
print(f"   Total skins: {len(skin_db)}")
print(f"   Skins where 3-day min was lower than ML prediction: {len(capped_skins)}")
print(f"   Percentage capped: {len(capped_skins) / len(skin_db) * 100:.2f}%")
print()

# Statistics
print("💰 Price Statistics:")
print(f"   Average market price: ${skin_db['mean_price_usd'].mean():.2f}")
print(f"   Average 3-day minimum: ${skin_db['min_price_3day'].mean():.2f}")
print(f"   Average ML safe price (with cap): ${skin_db['safe_price_usd_ML'].mean():.2f}")
print(f"   Average discount: {(1 - skin_db['safe_price_usd_ML'] / skin_db['mean_price_usd']).mean() * 100:.2f}%")
print()

# Create visualization
os.makedirs("../data/report/visualizations", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution of prices
ax = axes[0, 0]
ax.hist([skin_db['mean_price_usd'], skin_db['min_price_3day'], skin_db['safe_price_usd_ML']], 
        bins=50, alpha=0.7, label=['Market Price', '3-Day Min', 'Safe Price'])
ax.set_xlabel('Price (USD)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Price Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)  # Zoom in for better visibility

# 2. Scatter: Safe price vs Market price
ax = axes[0, 1]
scatter = ax.scatter(skin_db['mean_price_usd'], skin_db['safe_price_usd_ML'], 
                     alpha=0.3, s=10, c=skin_db['min_price_3day'], cmap='viridis')
ax.plot([0, skin_db['mean_price_usd'].max()], [0, skin_db['mean_price_usd'].max()], 
        'r--', alpha=0.3, label='Perfect Match')
ax.set_xlabel('Market Price (USD)', fontsize=11)
ax.set_ylabel('Safe Price (USD)', fontsize=11)
ax.set_title('Safe Price vs Market Price', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='3-Day Min')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scatter: Safe price vs 3-day min
ax = axes[1, 0]
ax.scatter(skin_db['min_price_3day'], skin_db['safe_price_usd_ML'], 
           alpha=0.3, s=10, c='purple')
ax.plot([0, skin_db['min_price_3day'].max()], [0, skin_db['min_price_3day'].max()], 
        'r--', alpha=0.5, label='Perfect Match')
ax.set_xlabel('3-Day Minimum Price (USD)', fontsize=11)
ax.set_ylabel('Safe Price (USD)', fontsize=11)
ax.set_title('Safe Price vs 3-Day Minimum', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Discount distribution
ax = axes[1, 1]
discount_pct = (1 - skin_db['safe_price_usd_ML'] / skin_db['mean_price_usd']) * 100
ax.hist(discount_pct, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax.axvline(discount_pct.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {discount_pct.mean():.2f}%')
ax.set_xlabel('Discount (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Discount Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../data/report/visualizations/3day_cap_analysis.png", dpi=300, bbox_inches='tight')
print("💾 Saved → ../data/report/visualizations/3day_cap_analysis.png")
plt.show()

# Top examples where cap was significant
print("\n📈 Top 20 Skins Most Affected by 3-Day Cap:")
print("=" * 80)
top_capped = skin_db.nlargest(20, 'min_price_3day')[['mean_price_usd', 'min_price_3day', 'safe_price_usd_ML']]
top_capped['market_vs_safe'] = (1 - top_capped['safe_price_usd_ML'] / top_capped['mean_price_usd']) * 100
top_capped.index = top_capped.index.str[:60]  # Truncate long names
print(top_capped.to_string())
print()

print("=" * 80)
print("ANALYSIS COMPLETE! 🎉")
print("=" * 80)

