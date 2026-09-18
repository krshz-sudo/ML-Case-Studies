"""
generate_data.py
-----------------
Synthesizes a realistic (but fully synthetic / privacy-safe) patient-encounter
dataset for the 30-day hospital readmission case study.

Unlike a fixed downloaded CSV, this generator lets anyone reproduce the full
pipeline end-to-end without needing external/proprietary hospital data, while
still reflecting realistic clinical relationships between vitals, prior
utilization, comorbidity burden and readmission risk.

Run:
    python src/generate_data.py --n 6000 --seed 42 --out data/patient_encounters.csv
"""

import argparse
import numpy as np
import pandas as pd

DIAGNOSES = [
    "Heart Failure", "COPD", "Pneumonia", "Diabetes Mellitus", "Sepsis",
    "Chronic Kidney Disease", "Stroke", "Hip Fracture", "Acute MI", "Cirrhosis",
]

# Base readmission risk multiplier per primary diagnosis (clinically-informed
# ordering: HF, sepsis and CKD tend to have higher 30-day readmission rates).
DIAGNOSIS_RISK = {
    "Heart Failure": 1.55, "Sepsis": 1.45, "Chronic Kidney Disease": 1.35,
    "COPD": 1.30, "Cirrhosis": 1.25, "Acute MI": 1.15, "Pneumonia": 1.05,
    "Stroke": 1.00, "Hip Fracture": 0.85, "Diabetes Mellitus": 0.95,
}

DISCHARGE_TO = ["Home", "Home Health Care", "Skilled Nursing Facility",
                "Rehabilitation Facility", "Hospice"]
INSURANCE = ["Private", "Medicare", "Medicaid", "Uninsured"]


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(64, 16, n), 18, 95).round().astype(int)
    gender = rng.choice(["Male", "Female"], n)
    primary_diagnosis = rng.choice(DIAGNOSES, n, p=_diag_probs())

    # --- Vitals at discharge (clinically plausible ranges + noise) ---
    heart_rate = np.clip(rng.normal(84, 14, n), 45, 160)
    systolic_bp = np.clip(rng.normal(128, 20, n), 80, 210)
    diastolic_bp = np.clip(rng.normal(78, 12, n), 45, 130)
    resp_rate = np.clip(rng.normal(18, 3, n), 10, 34)
    temp_c = np.clip(rng.normal(37.0, 0.5, n), 35.0, 40.0)
    spo2 = np.clip(rng.normal(96, 2.5, n), 82, 100)
    glucose = np.clip(rng.normal(120, 45, n), 60, 400)

    # --- Utilization history ---
    prior_admissions_12mo = rng.poisson(0.9, n)
    prior_ed_visits_12mo = rng.poisson(1.1, n)
    num_procedures = rng.poisson(2.0, n)
    days_in_hospital = np.clip(rng.gamma(2.2, 2.0, n), 1, 45).round().astype(int)
    num_medications = np.clip(rng.poisson(7, n), 0, 30)

    # --- Comorbidity burden (Charlson-like composite score) ---
    comorbidity_score = np.clip(rng.poisson(2.4, n), 0, 12)

    discharge_to = rng.choice(DISCHARGE_TO, n, p=[0.55, 0.18, 0.13, 0.10, 0.04])
    insurance_type = rng.choice(INSURANCE, n, p=[0.35, 0.40, 0.20, 0.05])
    follow_up_scheduled_7d = rng.choice([1, 0], n, p=[0.62, 0.38])

    # ---------------- Ground-truth risk model (latent, then thresholded) ----
    diag_mult = np.array([DIAGNOSIS_RISK[d] for d in primary_diagnosis])

    logit = (
        -3.4
        + 0.014 * (age - 64)
        + 0.55 * np.log1p(prior_admissions_12mo)
        + 0.35 * np.log1p(prior_ed_visits_12mo)
        + 0.16 * comorbidity_score
        + 0.05 * days_in_hospital
        + 0.03 * num_medications
        + 0.90 * (diag_mult - 1.0)
        + 0.012 * (heart_rate - 84)
        + 0.02 * (resp_rate - 18)
        - 0.05 * (spo2 - 96)
        + 0.004 * (glucose - 120)
        - 0.55 * follow_up_scheduled_7d
        + np.where(discharge_to == "Home", -0.25, 0.0)
        + np.where(discharge_to == "Hospice", -1.4, 0.0)
        + np.where(insurance_type == "Uninsured", 0.30, 0.0)
        + rng.normal(0, 0.55, n)  # unobserved patient/social factors
    )
    prob = 1 / (1 + np.exp(-logit))
    readmitted_30d = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "primary_diagnosis": primary_diagnosis,
        "heart_rate": heart_rate.round(1),
        "systolic_bp": systolic_bp.round(1),
        "diastolic_bp": diastolic_bp.round(1),
        "respiratory_rate": resp_rate.round(1),
        "temperature_c": temp_c.round(1),
        "spo2": spo2.round(1),
        "glucose_mg_dl": glucose.round(1),
        "prior_admissions_12mo": prior_admissions_12mo,
        "prior_ed_visits_12mo": prior_ed_visits_12mo,
        "num_procedures": num_procedures,
        "days_in_hospital": days_in_hospital,
        "num_medications": num_medications,
        "comorbidity_score": comorbidity_score,
        "discharge_to": discharge_to,
        "insurance_type": insurance_type,
        "follow_up_scheduled_7d": follow_up_scheduled_7d,
        "readmitted_30d": readmitted_30d,
    })

    # Inject a small amount of realistic missingness / duplicates for the
    # preprocessing step to handle.
    for col in ["glucose_mg_dl", "num_medications", "insurance_type"]:
        mask = rng.uniform(0, 1, n) < 0.02
        df.loc[mask, col] = np.nan
    dup_idx = rng.choice(df.index, size=max(1, n // 200), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


def _diag_probs():
    base = np.array([12, 11, 13, 14, 8, 9, 10, 9, 7, 7], dtype=float)
    return base / base.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/patient_encounters.csv")
    args = ap.parse_args()

    data = generate(args.n, args.seed)
    data.to_csv(args.out, index=False)
    print(f"Wrote {len(data)} rows to {args.out}")
    print(f"Readmission rate: {data['readmitted_30d'].mean():.3f}")
