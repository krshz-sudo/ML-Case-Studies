# Credit Card Fraud Detection

Flags fraudulent transactions in a highly imbalanced dataset (0.38% fraud), tuned with an operating threshold that reflects how many false alarms a review team can realistically handle.

![Pipeline](outputs/pipeline_diagram.svg)

## Problem

Fraud is rare, which means a model can look accurate while catching almost nothing. The real challenge here isn't training a classifier — it's evaluating and tuning it in a way that isn't fooled by the imbalance.

## Data

Synthetic transaction data (`data/transactions.csv`, 60,000 rows): 10 anonymized behavioral features (V1-V10), transaction amount, merchant category, channel, distance from home, distance from the last transaction, and ratio of the transaction amount to the cardholder's usual spend. Only ~70% of fraud cases carry an obvious signal, so the problem isn't trivially separable.

## Approach

1. **Clean + preprocess** — impute, `StandardScaler` on numeric columns, `OneHotEncoder` on categorical columns.
2. **75/25 stratified split** so both train and test keep the real 0.38% fraud rate.
3. **SMOTE on the training set only** — synthesizes new fraud examples by interpolating between real ones, rebalancing training data from 170 fraud / 44,830 legit to 44,830 / 44,830. The test set is left untouched.
4. **Two models trained and compared**: XGBoost (primary) vs. Random Forest (baseline).
5. **Evaluated on both ROC-AUC and PR-AUC** — ROC-AUC alone is misleading this imbalanced, PR-AUC exposes the real gap between models.
6. **Threshold tuned to a business rule**: lowest threshold that still keeps precision >= 30% (roughly 2 false alarms per real fraud caught), rather than the default 0.5.

## Tools

Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), matplotlib.

## Results

| Metric | XGBoost | Random Forest |
|---|---|---|
| ROC-AUC | 0.992 | 0.997 |
| PR-AUC | **0.930** | 0.766 |

ROC-AUC alone would suggest Random Forest is better — PR-AUC shows the opposite, which is why XGBoost was chosen.

**At the tuned threshold (0.0082):**
- Caught 55 of 57 fraud cases in the test set (~96.5% recall)
- 128 false alarms out of 14,943 legitimate transactions (~30% precision, meeting the target floor)

## Repo structure

```
src/generate_data.py   synthetic data generator
src/train.py            cleaning, SMOTE, training, evaluation, threshold search
data/                    generated dataset
notebooks/               exploratory analysis
outputs/                 metrics, trained model, plots
```

## Running it

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
```
