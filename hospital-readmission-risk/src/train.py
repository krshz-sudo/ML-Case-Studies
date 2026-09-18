"""
train.py
--------
End-to-end pipeline for the 30-day hospital readmission case study:

1. Load + clean the patient encounter data (drop duplicates, report missing).
2. Build a preprocessing pipeline: StandardScaler for numeric vitals/utilization
   features, OneHotEncoder for categorical fields.
3. Train two Logistic Regression variants:
     - L2-regularized (the primary model requested by the case study)
     - Unregularized (C very large) as a reference point
4. Evaluate both with ROC-AUC + classification report.
5. Perform a clinical cost analysis across decision thresholds:
     cost(FN) = avoidable readmission cost, cost(FP) = unnecessary
     post-discharge intervention cost, and pick the threshold that
     minimizes expected cost rather than defaulting to 0.5.
6. Save plots (ROC curve, cost-vs-threshold curve, coefficient importances)
   and a metrics.json summary to outputs/.

Run:
    python src/train.py --data data/patient_encounters.csv --outdir outputs
"""

import argparse
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

NUMERIC_FEATURES = [
    "age", "heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate",
    "temperature_c", "spo2", "glucose_mg_dl", "prior_admissions_12mo",
    "prior_ed_visits_12mo", "num_procedures", "days_in_hospital",
    "num_medications", "comorbidity_score", "follow_up_scheduled_7d",
]
CATEGORICAL_FEATURES = ["gender", "primary_diagnosis", "discharge_to", "insurance_type"]
TARGET = "readmitted_30d"

# Clinical cost assumptions (illustrative, documented in README):
# - Missing a true readmission (FN) costs more: the penalty/utilization cost
#   of an unplanned readmission plus the lost chance to intervene.
# - A false alarm (FP) costs the price of an unnecessary post-discharge
#   intervention (extra nurse call, home visit, follow-up appointment).
COST_FN = 5000.0
COST_FP = 500.0


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    missing = df.isna().sum()
    missing = missing[missing > 0]

    print(f"Loaded {n_before} rows, dropped {n_dupes} duplicates -> {len(df)} rows")
    if len(missing):
        print("Missing values by column:")
        print(missing.to_string())

    # Numeric missing -> median impute later inside pipeline is cleaner, but
    # for a categorical field like insurance_type we fill an explicit
    # "Unknown" category so OneHotEncoder has something well-defined.
    if "insurance_type" in df.columns:
        df["insurance_type"] = df["insurance_type"].fillna("Unknown")

    return df


def build_pipeline(C: float) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    # Numeric NaNs (glucose, num_medications) handled with simple median fill
    # via a small wrapper: StandardScaler can't handle NaN, so impute first.
    from sklearn.impute import SimpleImputer
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    clf = LogisticRegression(penalty="l2", C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("prep", preprocess), ("clf", clf)])


def expected_cost(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return fp * COST_FP + fn * COST_FN, tn, fp, fn, tp


def find_optimal_threshold(y_true, probs):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, probs, t)[0] for t in thresholds]
    best_idx = int(np.argmin(costs))
    return thresholds[best_idx], costs, thresholds


def main(data_path: str, outdir: str):
    df = load_and_clean(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    results = {}
    models = {}
    for name, C in [("l2_regularized", 1.0), ("unregularized", 1e6)]:
        pipe = build_pipeline(C=C)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        results[name] = {"roc_auc": auc}
        models[name] = (pipe, probs)
        print(f"\n=== {name} (C={C}) ===")
        print(f"ROC-AUC: {auc:.4f}")
        preds_default = (probs >= 0.5).astype(int)
        print(classification_report(y_test, preds_default, digits=3))

    # Use the L2-regularized model (the case study's requested approach) for
    # the clinical cost / threshold analysis and final artifacts.
    best_pipe, best_probs = models["l2_regularized"]
    best_threshold, costs, thresholds = find_optimal_threshold(y_test, best_probs)
    total_cost_default, *_ = expected_cost(y_test, best_probs, 0.5)
    total_cost_opt, tn, fp, fn, tp = expected_cost(y_test, best_probs, best_threshold)

    print(f"\nDefault threshold 0.50 -> expected cost ${total_cost_default:,.0f}")
    print(f"Cost-optimal threshold {best_threshold:.2f} -> expected cost "
          f"${total_cost_opt:,.0f}  (TN={tn}, FP={fp}, FN={fn}, TP={tp})")

    # ---------------- Plots ----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (pipe, probs) in models.items():
        RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)
    ax.set_title("ROC Curve — L2-regularized vs. Unregularized Logistic Regression")
    fig.tight_layout()
    fig.savefig(f"{outdir}/roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, costs, color="darkred")
    ax.axvline(best_threshold, ls="--", color="black",
               label=f"optimal threshold = {best_threshold:.2f}")
    ax.axvline(0.5, ls=":", color="gray", label="default threshold = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost ($)")
    ax.set_title("Expected Clinical Cost vs. Decision Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/cost_vs_threshold.png", dpi=150)
    plt.close(fig)

    # Coefficient importance (top |coef| for the L2 model)
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    coefs = best_pipe.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_names[i] for i in order][::-1], coefs[order][::-1],
            color=["#c0392b" if c > 0 else "#2980b9" for c in coefs[order][::-1]])
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title("Top 15 Predictors of 30-Day Readmission (L2 model)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/feature_importance.png", dpi=150)
    plt.close(fig)

    # Confusion matrices at both thresholds for the report
    cm_default = confusion_matrix(y_test, (best_probs >= 0.5).astype(int))
    cm_opt = confusion_matrix(y_test, (best_probs >= best_threshold).astype(int))

    metrics = {
        "roc_auc": results,
        "cost_assumptions": {"cost_false_negative": COST_FN, "cost_false_positive": COST_FP},
        "default_threshold_0.5": {
            "expected_cost": total_cost_default,
            "confusion_matrix": cm_default.tolist(),
        },
        "cost_optimal_threshold": {
            "threshold": best_threshold,
            "expected_cost": total_cost_opt,
            "confusion_matrix": cm_opt.tolist(),
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
        "readmission_rate_test": float(y_test.mean()),
    }
    with open(f"{outdir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(best_pipe, f"{outdir}/model_l2_logreg.joblib")
    print(f"\nSaved plots, metrics.json and model to {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/patient_encounters.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    main(args.data, args.outdir)
