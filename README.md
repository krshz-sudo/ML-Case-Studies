# ML Case Studies — Utkarsh Singh

Two end-to-end machine learning case studies, each covering the full pipeline from raw data to a tuned, business-aware decision — not just a trained model.

| Case study | Problem | Model | Key result |
|---|---|---|---|
| [Hospital Readmission Risk](hospital-readmission-risk/) | Predict 30-day patient readmission | L2-regularized logistic regression | Cost-optimal threshold (0.54) saves ~$10,500 in expected cost vs. the default 0.5 cutoff |
| [Credit Card Fraud Detection](credit-card-fraud-detection/) | Flag fraudulent transactions in a 0.38%-imbalanced dataset | XGBoost (vs. Random Forest baseline) + SMOTE | PR-AUC of 0.930 vs. 0.766 for the baseline; tuned threshold catches ~96.5% of fraud at ~30% precision |

Each folder is a fully self-contained project with its own README, source code, data, notebook, and output plots. Click into either one for the full write-up (problem, approach, tools, and results).

## Repo structure

```
hospital-readmission-risk/       logistic regression + cost-based threshold tuning
credit-card-fraud-detection/     XGBoost + SMOTE + PR-AUC-driven threshold tuning
```

## Common approach across both

Both projects follow the same underlying pipeline: clean the data, preprocess it into a model-ready format (scikit-learn `Pipeline` + `ColumnTransformer`), train and compare models, then go one step further than a raw accuracy/AUC number — translating the model's output into a threshold chosen for a real-world reason (cost, or an operational precision floor), rather than defaulting to 0.5.
