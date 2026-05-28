# CO2 Emission Prediction in Asia — ML & SHAP Analysis

> Graduation thesis project — Dokuz Eylül University, Department of Statistics (2026)

Predicting per capita CO2 emissions across 16 Asian countries (1990–2020) using 7 machine learning algorithms, SHAP interpretability analysis, and automated policy generation via Google Gemini LLM.

---

## Overview

**Dataset:** Annual panel data (n = 496) for 16 Asian countries from the World Bank Open Data Portal  
**Target:** Per capita CO2 emissions (tonnes/year)  
**Period:** 1990–2020 | Train: 1990–2015 | Test: 2016–2020  
**Countries:** Bangladesh, China, India, Indonesia, Iran, Iraq, Japan, South Korea, Malaysia, Nepal, Pakistan, Philippines, Saudi Arabia, Thailand, Turkey, Vietnam

---

## Methodology

### 1. Data preprocessing
- RobustScaler normalization (median/IQR-based, robust to outliers)
- Skewness analysis across all features
- Environmental Kuznets Curve (EKC) quadratic term derived: [ln(GDP)]²

### 2. Voting-Based Feature Selection (VBFS)
Three independent criteria voted on the top-K features:
- F-statistic (`f_regression`)
- Mutual information (`mutual_info_regression`)
- Random Forest importance scores

Features selected (≥2 votes): GDP per capita, urbanization rate, renewable energy share, forest cover, total population.

### 3. Cross-validation strategy
`GroupKFold (k=5)` — each fold holds out a different country group to prevent data leakage inherent in panel settings.

### 4. Models compared

| Model | Test R² | RMSE | Train-Test Gap |
|---|---|---|---|
| CatBoost | **0.93** | **0.99** | 0.046 ✓ |
| XGBoost | 0.88 | 1.33 | 0.091 △ |
| LightGBM | 0.87 | 1.36 | 0.097 △ |
| Random Forest | 0.82 | 1.59 | 0.122 ✗ |
| Linear Regression | 0.79 | 1.72 | 0.059 △ |
| SVR | 0.79 | 1.72 | 0.033 ✓ |
| MLP | 0.77 | 1.81 | 0.059 △ |

> Model selection criterion: Gap < 0.05 (overfitting-safe) + lowest RMSE → **CatBoost**

### 5. SHAP interpretability (CatBoost)
- Global feature importance ranking
- Effect direction analysis (beeswarm plot)
- Country-level SHAP comparison (stacked bar chart)

Key findings:
- Renewable energy share: strongest CO2-reducing factor
- Urbanization rate & GDP per capita: primary emission drivers
- EKC peak at ~$27,941 GDP per capita

### 6. LLM policy generation
Google Gemini (`gemini-2.5-flash`) automatically generated 3-point country-specific policy recommendations based on each country's top SHAP drivers.

---

## Project structure

```
├── bitirme.ipynb           # Main notebook (EDA → VBFS → modeling → SHAP → LLM)
├── data/
│   └── asia_panel.csv      # World Bank panel dataset (16 countries, 1990–2020)
├── figures/                # All generated plots
├── requirements.txt
└── README.md
```

---

## Results summary

- CatBoost achieved **R² = 0.93**, **RMSE = 0.99 t/capita** on unseen test data (2016–2020)
- Train-Test R² gap of **0.046** confirms generalization without overfitting
- Renewable energy is the most impactful lever for CO2 reduction across Asia
- Country heterogeneity confirmed: single-policy approaches are insufficient for the region

---

## Interactive dashboard (Streamlit)

```bash
git clone https://github.com/ibrahimSalihreisoglu/CO2-Emission-Prediction-Asia.git
cd CO2-Emission-Prediction-Asia
pip install -r requirements.txt
streamlit run app.py
```

Features: EDA explorer, model results, CO2 prediction form (adjustable sliders → CatBoost inference)

---

## Tech stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `SHAP` `Streamlit` `Google Gemini API`

---

## Authors

Ibrahim Salihreisoğlu · Cenker Doğru · İrem Gençer · Büke Gedik  
Supervisor: Prof. Selma Gürler — Dokuz Eylül University, Faculty of Science
