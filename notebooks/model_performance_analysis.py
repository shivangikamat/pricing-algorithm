#!/usr/bin/env python3
"""
Model Performance Analysis
Calculates comprehensive statistics for all models to determine the best performer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 80)
print()

# Load the unified comparison
comparison_path = "../data/report/all_models_comparison.csv"
comparison_df = pd.read_csv(comparison_path, index_col=0)

print(f"✅ Loaded comparison data: {len(comparison_df)} skins")
print()

# Get all model columns
safe_price_cols = [col for col in comparison_df.columns if col.startswith('safe_price_')]
model_names = [col.replace('safe_price_', '') for col in safe_price_cols]

print(f"📊 Analyzing {len(model_names)} models:")
for name in model_names:
    print(f"   - {name}")
print()

# Calculate comprehensive statistics for each model
stats_summary = []

for col, model_name in zip(safe_price_cols, model_names):
    if col not in comparison_df.columns:
        continue
    
    # Get valid data (non-null, non-zero prices)
    valid_mask = (comparison_df[col].notna()) & (comparison_df['mean_price_usd'] > 0)
    valid_data = comparison_df[valid_mask]
    
    if len(valid_data) == 0:
        continue
    
    safe_prices = valid_data[col].values
    mean_prices = valid_data['mean_price_usd'].values
    
    # Calculate discounts
    discounts = (1 - safe_prices / mean_prices) * 100
    
    # Basic statistics
    mean_discount = discounts.mean()
    median_discount = np.median(discounts)
    std_discount = discounts.std()
    
    # Range statistics
    min_discount = discounts.min()
    max_discount = discounts.max()
    q25_discount = np.percentile(discounts, 25)
    q75_discount = np.percentile(discounts, 75)
    
    # Price statistics
    avg_safe_price = safe_prices.mean()
    total_discount_amount = (mean_prices - safe_prices).sum()
    
    # Calculate risk classification if available
    risk_col = f'risk_score_{model_name}'
    risk_accuracy = None
    if risk_col in comparison_df.columns:
        risk_scores = valid_data[risk_col].values
        # Assuming risky = 1 means high discount
        risky_mask = risk_scores >= 0.5
        risky_discount_avg = discounts[risky_mask].mean() if risky_mask.sum() > 0 else 0
        safe_discount_avg = discounts[~risky_mask].mean() if (~risky_mask).sum() > 0 else 0
        risk_separation = risky_discount_avg - safe_discount_avg
    else:
        risky_discount_avg = None
        safe_discount_avg = None
        risk_separation = None
    
    # Calculate variation coefficient (CV)
    cv = (std_discount / mean_discount) * 100 if mean_discount != 0 else 0
    
    # Calculate how often model applies discount (non-zero)
    discount_applied_rate = (discounts > 0.1).sum() / len(discounts) * 100
    
    # Aggregate statistics
    stats = {
        'Model': model_name,
        'Avg Discount (%)': round(mean_discount, 2),
        'Median Discount (%)': round(median_discount, 2),
        'Std Discount (%)': round(std_discount, 2),
        'CV (%)': round(cv, 2),
        'Min Discount (%)': round(min_discount, 2),
        'Max Discount (%)': round(max_discount, 2),
        'Q25 (%)': round(q25_discount, 2),
        'Q75 (%)': round(q75_discount, 2),
        'Avg Safe Price ($)': round(avg_safe_price, 2),
        'Total Discount ($)': round(total_discount_amount, 0),
        'Discount Applied Rate (%)': round(discount_applied_rate, 2),
        'Samples': len(valid_data)
    }
    
    if risk_separation is not None:
        stats['Risk Separation'] = round(risk_separation, 2)
    
    stats_summary.append(stats)

# Create summary dataframe
stats_df = pd.DataFrame(stats_summary)

print("=" * 80)
print("COMPREHENSIVE MODEL STATISTICS")
print("=" * 80)
print()

# Display formatted statistics
for col in ['Model', 'Avg Discount (%)', 'Std Discount (%)', 'CV (%)', 'Risk Separation', 'Samples']:
    if col in stats_df.columns:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(stats_df[[col]].to_string(index=False))
        print()

# Create detailed statistics table
print("=" * 80)
print("DETAILED STATISTICS TABLE")
print("=" * 80)
print()

# Reorder columns for better readability
display_cols = ['Model', 'Samples', 'Avg Discount (%)', 'Median Discount (%)', 'Std Discount (%)', 
                'CV (%)', 'Discount Applied Rate (%)', 'Risk Separation']
display_cols = [col for col in display_cols if col in stats_df.columns]
print(stats_df[display_cols].to_string(index=False))
print()

# Save statistics
stats_output_path = "../data/report/model_performance_stats.csv"
stats_df.to_csv(stats_output_path, index=False)
print(f"💾 Saved statistics → {stats_output_path}")
print()

# Ranking Analysis
print("=" * 80)
print("MODEL RANKINGS")
print("=" * 80)
print()

# Rank by different criteria
rankings = {}

# 1. Most conservative (highest avg discount)
rankings['Most Conservative'] = stats_df.nlargest(1, 'Avg Discount (%)')[['Model', 'Avg Discount (%)']].values[0]

# 2. Most consistent (lowest CV)
rankings['Most Consistent'] = stats_df.nsmallest(1, 'CV (%)')[['Model', 'CV (%)']].values[0]

# 3. Best risk separation (highest risk separation)
if 'Risk Separation' in stats_df.columns:
    rankings['Best Risk Separation'] = stats_df.nlargest(1, 'Risk Separation')[['Model', 'Risk Separation']].values[0]

# 4. Most aggressive (lowest avg discount)
rankings['Most Aggressive'] = stats_df.nsmallest(1, 'Avg Discount (%)')[['Model', 'Avg Discount (%)']].values[0]

# 5. Most selective (lowest discount applied rate)
rankings['Most Selective'] = stats_df.nsmallest(1, 'Discount Applied Rate (%)')[['Model', 'Discount Applied Rate (%)']].values[0]

# Display rankings
for category, data in rankings.items():
    model, value = data
    print(f"🏆 {category}:")
    print(f"   {model} ({value})")
    print()

# Create visualizations
print("=" * 80)
print("Generating visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Average Discount Comparison
ax1 = axes[0, 0]
sorted_models = stats_df.sort_values('Avg Discount (%)', ascending=False)
ax1.barh(range(len(sorted_models)), sorted_models['Avg Discount (%)'], color='steelblue')
ax1.set_yticks(range(len(sorted_models)))
ax1.set_yticklabels(sorted_models['Model'])
ax1.set_xlabel('Average Discount (%)', fontsize=11)
ax1.set_title('Average Discount by Model', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(sorted_models['Avg Discount (%)']):
    ax1.text(v + 0.5, i, f'{v:.2f}%', va='center')

# 2. Consistency (CV) Comparison
ax2 = axes[0, 1]
sorted_models_cv = stats_df.sort_values('CV (%)', ascending=True)
ax2.barh(range(len(sorted_models_cv)), sorted_models_cv['CV (%)'], color='coral')
ax2.set_yticks(range(len(sorted_models_cv)))
ax2.set_yticklabels(sorted_models_cv['Model'])
ax2.set_xlabel('Coefficient of Variation (%)', fontsize=11)
ax2.set_title('Consistency (Lower CV = More Consistent)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(sorted_models_cv['CV (%)']):
    ax2.text(v + 1, i, f'{v:.2f}%', va='center')

# 3. Discount Applied Rate
ax3 = axes[1, 0]
sorted_models_rate = stats_df.sort_values('Discount Applied Rate (%)', ascending=False)
ax3.barh(range(len(sorted_models_rate)), sorted_models_rate['Discount Applied Rate (%)'], color='green')
ax3.set_yticks(range(len(sorted_models_rate)))
ax3.set_yticklabels(sorted_models_rate['Model'])
ax3.set_xlabel('Discount Applied Rate (%)', fontsize=11)
ax3.set_title('How Often Model Applies Discount', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(sorted_models_rate['Discount Applied Rate (%)']):
    ax3.text(v + 1, i, f'{v:.1f}%', va='center')

# 4. Risk Separation
ax4 = axes[1, 1]
if 'Risk Separation' in stats_df.columns:
    risk_models = stats_df[stats_df['Risk Separation'].notna()]
    sorted_risk = risk_models.sort_values('Risk Separation', ascending=False)
    ax4.barh(range(len(sorted_risk)), sorted_risk['Risk Separation'], color='purple')
    ax4.set_yticks(range(len(sorted_risk)))
    ax4.set_yticklabels(sorted_risk['Model'])
    ax4.set_xlabel('Risk Separation (Risky Avg - Safe Avg)', fontsize=11)
    ax4.set_title('Risk Differentiation Ability', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(sorted_risk['Risk Separation']):
        ax4.text(v + 0.5, i, f'{v:.2f}%', va='center')
else:
    ax4.text(0.5, 0.5, 'Risk data not available', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    ax4.set_title('Risk Separation', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save visualization
os.makedirs("../data/report/visualizations", exist_ok=True)
viz_path = "../data/report/visualizations/model_performance_analysis.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"💾 Saved visualization → {viz_path}")
plt.show()

# Overall Winner
print()
print("=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
print()

# Calculate composite score (weighted average)
stats_df['Conservative_Score'] = (stats_df['Avg Discount (%)'] / stats_df['Avg Discount (%)'].max()) * 100
stats_df['Consistency_Score'] = ((stats_df['CV (%)'].max() - stats_df['CV (%)']) / stats_df['CV (%)'].max()) * 100
stats_df['Selectivity_Score'] = (1 - stats_df['Discount Applied Rate (%)'] / 100) * 100

if 'Risk Separation' in stats_df.columns:
    risk_models = stats_df[stats_df['Risk Separation'].notna()]
    if len(risk_models) > 0:
        stats_df.loc[risk_models.index, 'Risk_Score'] = (risk_models['Risk Separation'] / risk_models['Risk Separation'].max()) * 100
    else:
        stats_df['Risk_Score'] = 0
else:
    stats_df['Risk_Score'] = 0

# Composite score (can weight each component)
stats_df['Composite_Score'] = (
    stats_df['Conservative_Score'] * 0.3 +
    stats_df['Consistency_Score'] * 0.3 +
    stats_df['Selectivity_Score'] * 0.2 +
    stats_df['Risk_Score'] * 0.2
)

# Find winner
winner = stats_df.nlargest(1, 'Composite_Score')[['Model', 'Composite_Score', 'Avg Discount (%)', 'CV (%)']].iloc[0]

print(f"🏆 BEST OVERALL MODEL: {winner['Model']}")
print(f"   Composite Score: {winner['Composite_Score']:.2f}/100")
print(f"   Avg Discount: {winner['Avg Discount (%)']:.2f}%")
print(f"   Consistency: {winner['CV (%)']:.2f}% CV")
print()

print("Score Breakdown:")
print(f"   Conservative (30%): Higher discount = safer")
print(f"   Consistency (30%): Lower variation = more reliable")
print(f"   Selectivity (20%): Only discount when needed")
print(f"   Risk Separation (20%): Better risky vs safe distinction")
print()

# Save ranked summary
ranked_output_path = "../data/report/model_rankings.csv"
stats_df_sorted = stats_df.sort_values('Composite_Score', ascending=False)
stats_df_sorted[['Model', 'Composite_Score', 'Avg Discount (%)', 'CV (%)', 'Discount Applied Rate (%)', 'Risk Separation']].to_csv(ranked_output_path, index=False)
print(f"💾 Saved rankings → {ranked_output_path}")
print()

print("=" * 80)
print("ANALYSIS COMPLETE! 🎉")
print("=" * 80)

