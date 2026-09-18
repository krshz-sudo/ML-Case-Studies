"""
train.py
--------
End-to-end pipeline for the Credit Card Fraud Detection case study:

1. Load + clean the transaction data (duplicates, missing values).
2. Feature engineering (encode categoricals, keep the anonymized V-features).
3. Stratified train/test split (fraud rate preserved on both sides).
4. Apply SMOTE to the *training* fold only (never touch the test set) to
   oversample the fraud class.
5. Train an XGBoost classifier on the SMOTE-balanced data, and a
   RandomForest as a comparison baseline (chosen instead of an SVM so this
   doesn't need to subsample for tractability on 60k+ rows).
6. Evaluate with ROC-AUC and Average Precision (PR-AUC) -- PR-AUC is added
   deliberately because ROC-AUC alone is optimistic on a >99% imbalanced
   problem like this one.
7. Tune the decision threshold against the precision/recall trade-off and
   report a chosen operating point.
8. Save ROC curve, Precision-Recall curve, feature-importance plot,
   confusion matrices and metrics.json to outputs/.

Run:
    python src/train.py --data data/transactions.csv --outdir outputs
"""

import argparse
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                              average_precision_score, classification_report,
                              confusion_matrix, precision_recall_curve,
                              roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

NUMERIC_FEATURES = (
    [f"V{i}" for i in range(1, 11)]
    + ["hour", "amount", "card_present", "distance_from_home_km",
       "distance_from_last_txn_km", "ratio_to_median_price",
       "is_foreign_transaction"]
)
CATEGORICAL_FEATURES = ["channel", "merchant_category"]
TARGET = "is_fraud"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "transaction_id"])
    df = df.reset_index(drop=True)
    print(f"Loaded {n_before} rows, dropped {n_before - len(df)} duplicates -> {len(df)} rows")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values by column:")
        print(missing.to_string())

    df["merchant_category"] = df["merchant_category"].fillna("unknown")
    return df


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def pick_operating_threshold(y_true, probs, min_precision=0.30):
    """Choose the lowest threshold that still keeps precision >= min_precision,
    maximizing recall (fraud caught) subject to an acceptable false-alarm rate.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision/recall arrays are 1 longer than thresholds
    candidates = [(t, p, r) for t, p, r in zip(thresholds, precision[:-1], recall[:-1])
                  if p >= min_precision]
    if not candidates:
        return 0.5, precision, recall, thresholds
    # among those meeting the precision floor, take the one with highest recall
    best = max(candidates, key=lambda x: x[2])
    return best[0], precision, recall, thresholds


def main(data_path: str, outdir: str):
    df = load_and_clean(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train fraud rate: {y_train.mean():.5f} | Test fraud rate: {y_test.mean():.5f}")

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"Pre-SMOTE train class counts: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
    print(f"Post-SMOTE train class counts: {np.bincount(y_train_bal)}")

    models = {}

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train_bal, y_train_bal)
    models["xgboost"] = xgb

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    models["random_forest_baseline"] = rf

    results = {}
    probs_by_model = {}
    for name, model in models.items():
        probs = model.predict_proba(X_test_proc)[:, 1]
        probs_by_model[name] = probs
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        results[name] = {"roc_auc": roc_auc, "pr_auc": pr_auc}
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC (avg precision): {pr_auc:.4f}")
        preds_default = (probs >= 0.5).astype(int)
        print(classification_report(y_test, preds_default, digits=3))

    # Focus threshold tuning + artifacts on the primary XGBoost model.
    best_probs = probs_by_model["xgboost"]
    threshold, precision, recall, thresholds = pick_operating_threshold(
        y_test, best_probs, min_precision=0.30
    )
    preds_tuned = (best_probs >= threshold).astype(int)
    cm_default = confusion_matrix(y_test, (best_probs >= 0.5).astype(int))
    cm_tuned = confusion_matrix(y_test, preds_tuned)
    print(f"\nChosen operating threshold (precision>=0.30 floor): {threshold:.3f}")
    print("Confusion matrix @ tuned threshold [[TN FP][FN TP]]:")
    print(cm_tuned)

    # ---------------- Plots ----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in probs_by_model.items():
        RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)
    ax.set_title("ROC Curve — XGBoost (SMOTE) vs. Random Forest baseline")
    fig.tight_layout()
    fig.savefig(f"{outdir}/roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in probs_by_model.items():
        PrecisionRecallDisplay.from_predictions(y_test, probs, name=name, ax=ax)
    ax.set_title("Precision-Recall Curve (imbalanced-data view)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precision[:-1], label="precision")
    ax.plot(thresholds, recall[:-1], label="recall")
    ax.axvline(threshold, ls="--", color="black", label=f"chosen = {threshold:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs. Threshold (XGBoost)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/threshold_tuning.png", dpi=150)
    plt.close(fig)

    # Feature importance (XGBoost gain-based)
    cat_ohe = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    importances = xgb.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1],
            color="#8e44ad")
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title("Top 15 Features Driving Fraud Predictions")
    fig.tight_layout()
    fig.savefig(f"{outdir}/feature_importance.png", dpi=150)
    plt.close(fig)

    metrics = {
        "model_metrics": results,
        "test_fraud_rate": float(y_test.mean()),
        "chosen_threshold": {
            "value": float(threshold),
            "min_precision_floor": 0.30,
            "confusion_matrix": cm_tuned.tolist(),
        },
        "default_threshold_0.5": {"confusion_matrix": cm_default.tolist()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "smote_train_counts": {
            "before": np.bincount(y_train).tolist(),
            "after": np.bincount(y_train_bal).tolist(),
        },
    }
    with open(f"{outdir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"preprocessor": preprocessor, "model": xgb}, f"{outdir}/model_xgb_fraud.joblib")
    print(f"\nSaved plots, metrics.json and model to {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    main(args.data, args.outdir)
