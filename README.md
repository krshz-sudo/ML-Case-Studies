# ML Case Studies

**Utkarsh Singh**

Two machine learning case studies I built end to end: cleaning raw data, training and comparing models, and then tuning a decision threshold for a reason that actually makes sense for the problem instead of just leaving it at the default 0.5.

| Case study | Problem | Model used | Result |
|---|---|---|---|
| [Hospital Readmission Risk](hospital-readmission-risk/) | Predict if a patient gets readmitted within 30 days | L2 regularized logistic regression | Threshold tuned by cost, saves about $10,500 in expected cost on the test set |
| [Credit Card Fraud Detection](credit-card-fraud-detection/) | Catch fraud in a dataset where only 0.38% of transactions are fraud | XGBoost, compared against a Random Forest baseline | PR-AUC of 0.93 vs 0.77 for the baseline, tuned threshold catches ~96.5% of fraud |

Open either folder for the full write up: the problem, what I tried, the tools used, and the actual numbers.

## Repo structure

```
hospital-readmission-risk/       logistic regression, cost based threshold tuning
credit-card-fraud-detection/     XGBoost, SMOTE, PR-AUC driven threshold tuning
```

## What's common between the two

Both projects follow the same basic shape: clean the data, build a preprocessing pipeline (scikit-learn `Pipeline` and `ColumnTransformer`), train and compare a couple of models, then don't stop at reporting AUC. In both cases I picked the classification threshold based on what a real mistake would actually cost, instead of defaulting to 0.5 like most beginner projects do.
