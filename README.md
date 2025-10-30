# 🎯 SkinSafe Pricing — Detecting Manipulated Skins & Predicting Safe Prices

This project builds an **intelligent pricing system for game skins** (e.g. CS2 items), using market data from **Buff163** via the [Pricempire API](https://pricempire.com/).  
It identifies **price manipulation patterns** and computes a **safe purchase price** for each skin using statistical analysis and multiple machine learning models.

---

## 🚀 Project Overview

### 🧠 Problem
Certain criminal organizations artificially inflate (“pump and dump”) skin prices, leading to huge losses when marketplaces correct.  
Skins with suspicious price spikes or liquidity patterns must be detected and priced safely.

### 🎯 Goal
Build a model that:
1. Detects **manipulated or volatile skins** using price data.
2. Calculates a **“safe buy” price** to minimize loss risk.
3. Compares multiple models (LDA, SVM, Random Forest, BGMM, etc.) for accuracy and reliability.

---