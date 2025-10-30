# Best Model Analysis Report

## Executive Summary

After analyzing 5 machine learning models across 19,264 skins, here are the key findings:

### 🏆 Overall Winner: **BGMM** (Bayesian Gaussian Mixture Model)

**Composite Score: 53.23/100**

## Model Comparison

| Model | Avg Discount | Consistency (CV) | Risk Separation | Composite Score | Best For |
|-------|--------------|------------------|-----------------|-----------------|----------|
| **BGMM** | 0.93% | 97.06% | **24.07%** | **53.23** | Risk detection |
| Logistic Regression | 4.91% | 164.41% | 21.17% | 47.82 | Baseline comparison |
| SVM | 4.68% | 1108.11% | 16.95% | 27.15 | Classification |
| LDA | -6.83% | -533.89% | -40.40% | 15.98 | Not recommended |
| **Random Forest** | **31.44%** | **97.79%** | N/A | N/A | Conservative pricing |

## Detailed Analysis

### 1. Most Conservative: Random Forest
- **Avg Discount: 31.44%**
- Applies discount to 93.6% of skins
- Total discount amount: $1,958,543
- **Use Case**: When you want maximum protection against price drops
- **Cons**: Very aggressive discounting

### 2. Best Risk Separation: BGMM
- **Risk Separation: 24.07%**
- Distinguishes well between risky and safe skins
- **Use Case**: When you need to identify which skins are actually risky
- **Pros**: Unsupervised learning, no labels needed

### 3. Most Consistent: LDA
- **CV: -533.89%** (negative indicates high variation in practice)
- However, LDA has major issues (negative discounts, poor risk separation)
- **Recommendation**: Do not use LDA model

### 4. Most Selective: LDA
- **Discount Applied Rate: 41.95%**
- Only discounts when truly needed
- But negative discounts make this problematic

### 5. Most Balanced: BGMM
- Good risk separation (24.07%)
- Reasonable consistency (97.06% CV)
- Moderate discount rate (75.13%)
- **Best overall choice**

## Model-Specific Analysis

### BGMM (Bayesian Gaussian Mixture Model)
**Score: 53.23/100 ⭐⭐⭐⭐⭐**

Strengths:
- ✅ Best risk separation (24.07%)
- ✅ Good consistency (97.06% CV)
- ✅ Unsupervised (no training labels needed)
- ✅ Finds hidden patterns in data
- ✅ Balanced approach

Weaknesses:
- ⚠️ Lower average discount (0.93%)
- ⚠️ May be too conservative

**Use Case**: Main recommendation for balanced risk assessment

---

### Logistic Regression
**Score: 47.82/100 ⭐⭐⭐⭐**

Strengths:
- ✅ Good baseline for comparison
- ✅ Probability-based predictions
- ✅ Interpretable results
- ✅ Decent risk separation (21.17%)

Weaknesses:
- ⚠️ Moderate inconsistency (164.41% CV)
- ⚠️ Applies discount to all skins (100%)
- ⚠️ Linear decision boundary

**Use Case**: Reliable baseline, good for comparison

---

### Random Forest (ML Optimized)
**Score: N/A (incomplete risk scoring) ⭐⭐⭐⭐**

Strengths:
- ✅ **Most conservative** (31.44% avg discount)
- ✅ Excellent consistency (97.79% CV)
- ✅ Highest total protection ($1.96M saved)
- ✅ Data-driven, learned from 19K+ skins
- ✅ Includes 3-day minimum price cap

Weaknesses:
- ⚠️ Very aggressive discounting
- ⚠️ No risk score component in current implementation
- ⚠️ May over-discount safe skins

**Use Case**: Maximum price protection when safety is critical

---

### SVM (Support Vector Machine)
**Score: 27.15/100 ⭐⭐⭐**

Strengths:
- ✅ Non-linear pattern detection
- ✅ Some risk separation (16.95%)

Weaknesses:
- ⚠️ High variability (1108.11% CV)
- ⚠️ Inconsistent predictions
- ⚠️ Moderate risk separation

**Use Case**: Classification tasks, but other models perform better

---

### LDA (Linear Discriminant Analysis)
**Score: 15.98/100 ⭐**

Strengths:
- ✅ Selective (41.95% discount rate)

Weaknesses:
- ⚠️ **Negative average discount (-6.83%)** - prices ABOVE market!
- ⚠️ Poor risk separation (-40.40%)
- ⚠️ High variability (-533.89% CV)
- ⚠️ **Not recommended for use**

**Use Case**: None - do not use this model

---

## Key Findings

### 1. Conservatism Trade-off
- Random Forest offers maximum protection (31.44% discount) but may be too aggressive
- BGMM offers balanced protection (0.93% discount) with better risk detection
- Logistic Regression is middle ground (4.91% discount)

### 2. Consistency Matters
- Random Forest (97.79% CV) and BGMM (97.06% CV) are most reliable
- SVM (1108.11% CV) has high variability - unpredictable
- LDA has major issues with negative CV

### 3. Risk Assessment Quality
- BGMM best at distinguishing risky vs safe (24.07% separation)
- Logistic Regression second best (21.17% separation)
- LDA actually has negative separation (-40.40%) - worse than random

### 4. Practical Application
- **For balanced approach**: Use BGMM
- **For maximum safety**: Use Random Forest (with 3-day cap)
- **For comparison**: Use Logistic Regression
- **Avoid**: LDA (negative discounts, poor performance)

## Recommendations

### Primary Recommendation: **BGMM + Random Forest Hybrid**

Use a two-stage approach:
1. **BGMM** to identify risky skins
2. **Random Forest** for conservative pricing on risky skins only

This combines:
- Best risk detection (BGMM)
- Maximum protection (Random Forest)
- Balanced approach overall

### Alternative: Use Random Forest with 3-Day Cap

Since Random Forest has:
- 3-day minimum price cap protection
- Highest consistency
- Most conservative pricing
- Data-driven from large dataset

It's the safest overall choice if you want maximum protection.

### Implementation Priority

1. **Random Forest** - Already integrated with 3-day cap ✅
2. **BGMM** - Available for risk detection
3. **Logistic Regression** - Good baseline
4. **SVM** - Classification only
5. **LDA** - Do not use

## Conclusion

While Random Forest offers the most conservative pricing, **BGMM is the best balanced model** considering:
- Risk separation quality
- Consistency
- Practical applicability

For maximum protection, use **Random Forest with the 3-day cap** as it's the safest option.

## Data Quality Notes

- Analysis based on 19,264 valid skin samples
- Random Forest missing risk separation score in current implementation
- LDA shows concerning negative discounts indicating model issues
- All models trained on same feature set for fairness

## Next Steps

1. Consider implementing risk scoring for Random Forest
2. Remove or fix LDA model (negative discounts problematic)
3. Create hybrid model combining BGMM risk + Random Forest pricing
4. Implement automated model selection based on user preference (conservative vs balanced)

---

**Report Generated**: 2025-10-30  
**Models Analyzed**: 5  
**Samples**: 19,264 skins  
**Primary Recommendation**: BGMM for balanced approach, Random Forest for maximum safety

