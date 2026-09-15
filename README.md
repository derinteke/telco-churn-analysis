# 📉 Telco Customer Churn — Analysis & Prediction

Predict which telecom customers are likely to **churn** (cancel their service) and
explain *why*, so the business can target retention offers where they matter most.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-red)
![LightGBM](https://img.shields.io/badge/LightGBM-tuned-green)
![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Problem

A telecom company loses revenue every time a customer leaves, and keeping an existing
customer is far cheaper than acquiring a new one. This project builds an interpretable,
tuned churn model on the IBM Telco dataset (7,043 customers) and turns the findings into
concrete retention actions.

Only **~27% of customers churn**, so this is an **imbalanced** problem. Plain accuracy is
misleading, so I evaluate on **ROC-AUC, PR-AUC, recall and F1**, with class weighting to
prioritise catching churners.

## 📊 Results

Tuned with **Optuna** (5-fold cross-validation on ROC-AUC), evaluated on a held-out 20%
test set:

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|:-------:|:------:|:---------:|:--:|
| Logistic Regression (baseline) | 0.846 | 0.783 | 0.505 | 0.614 |
| LightGBM (tuned) | 0.848 | 0.805 | 0.515 | 0.628 |
| **XGBoost (tuned)** | **0.850** | 0.797 | 0.515 | 0.625 |

**Best model: XGBoost — holdout ROC-AUC 0.850, PR-AUC 0.667.** At the default 0.50
threshold it catches ~80% of churners; the F1-optimal threshold (~0.59) trades some recall
for precision.

<p align="center">
  <img src="reports/figures/roc_curves.png" width="420">
  <img src="reports/figures/shap_importance.png" width="420">
</p>

### What actually moved the needle (honest write-up)

I ran a broad search (LightGBM / XGBoost / CatBoost, 40–150 Optuna trials each) and found
this dataset tops out around **~0.85 ROC-AUC**. The useful lessons:

- **Hyperparameter tuning was the biggest lever** (~+0.013 over the untuned booster);
  feature engineering added a little more.
- **Model choice barely mattered** — LightGBM, XGBoost and CatBoost land within 0.001.
- **SMOTE hurt** (0.835 vs 0.837 for `scale_pos_weight`), and a 3-model **ensemble did not
  beat the single best model**.

I think showing *where the ceiling is* and *what doesn't work* is more honest — and more
useful — than chasing a fake accuracy jump.

## 🔑 Key findings

- **Contract type is the strongest driver.** Month-to-month customers churn at **~43%**
  vs **~3%** for two-year contracts.
- **Churn peaks early** — most churn happens in the first months of tenure.
- **Missing add-ons hurt** — customers without *online security* or *tech support* churn more.
- **Payment friction matters** — electronic-check users churn more than auto-pay users.

<p align="center">
  <img src="reports/figures/eda_churn_by_contract.png" width="430">
  <img src="reports/figures/eda_tenure_distribution.png" width="430">
</p>

## 💡 Business recommendations

| Finding | Suggested action |
|---|---|
| Month-to-month churn ~43% | Incentivise 1–2 year contracts with a switching discount |
| Churn peaks in first months | Onboarding programme + early check-ins for new customers |
| No security / tech support → more churn | Bundle or offer free trials of these services |
| Electronic-check users churn more | Nudge customers toward automatic payments |

## 🗂️ Project structure

```
telco-churn-analysis/
├── data/
│   └── raw/telco_churn.csv        # source dataset
├── notebooks/
│   └── 01_churn_analysis.ipynb    # full story (bilingual TR/EN): EDA → tuning → SHAP
├── src/                           # reusable, importable modules
│   ├── config.py                  # paths, columns, tuned hyperparameters
│   ├── data.py                    # load + clean
│   ├── features.py                # feature engineering + preprocessing pipeline
│   ├── model.py                   # LogReg + tuned LightGBM + tuned XGBoost
│   ├── tune.py                    # Optuna hyperparameter search (reproducible)
│   ├── evaluate.py                # metrics & plots
│   ├── eda.py                     # EDA figure generation
│   └── explain.py                 # SHAP explainability
├── models/                        # saved best model + metrics.json (generated)
├── reports/figures/               # all figures (generated)
├── train.py                       # end-to-end: clean → engineer → train → evaluate
├── requirements.txt
└── README.md
```

## 🚀 Getting started

```bash
git clone <your-repo-url>
cd telco-churn-analysis

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train & evaluate all three models (writes models/ and reports/figures/)
python train.py

# Reproduce the hyperparameter search
python -m src.tune --model xgboost --trials 50

# Regenerate EDA / SHAP figures
python -m src.eda
python -m src.explain

# Explore the notebook
jupyter notebook notebooks/01_churn_analysis.ipynb
```

## 🛠️ Methodology

1. **Cleaning** — fix `TotalCharges` (text with blanks for tenure-0 customers) and make
   `SeniorCitizen` categorical.
2. **Feature engineering** — number of services, tenure bands, spend-per-tenure, internet
   flag, monthly/total ratio.
3. **Preprocessing** — a scikit-learn `ColumnTransformer` (scaling + one-hot) lives *inside*
   each `Pipeline`, so there is **no data leakage** and the saved model accepts raw input.
4. **Modeling & tuning** — LogReg baseline + LightGBM/XGBoost tuned with **Optuna** under
   5-fold CV; class imbalance handled with `scale_pos_weight` / `class_weight`.
5. **Evaluation** — stratified split, ROC-AUC / PR-AUC / recall / F1, confusion matrices,
   ROC curves, and **decision-threshold tuning**.
6. **Explainability** — SHAP values expose global feature importance.

## 🔭 Possible extensions

- Cost-sensitive threshold selection driven by campaign budget and customer lifetime value.
- Deployment as a FastAPI service + Streamlit dashboard, containerised with Docker.

## 📄 License

MIT — feel free to use and adapt.
