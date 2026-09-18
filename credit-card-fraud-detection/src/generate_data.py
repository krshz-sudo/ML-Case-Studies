"""
generate_data.py
-----------------
Synthesizes a heavily-imbalanced credit-card transaction dataset in the
spirit of the classic anonymized-PCA-feature fraud datasets (e.g. Kaggle's
`creditcardfraud` / IEEE-CIS), without needing to download a large
proprietary CSV. Anyone can regenerate the exact same benchmark locally.

Design choices that go beyond a bare PCA-feature dump:
  - 10 anonymized "PCA-style" continuous features (V1..V10), some of which
    carry a genuine (but noisy) fraud signal, mimicking the real dataset's
    behavior without being derived from it.
  - Realistic engineered context features: hour of day, merchant category,
    transaction channel, card-present flag, distance from home, distance
    from the previous transaction, and ratio to the cardholder's median
    purchase price -- the kind of features a real fraud team would build.
  - A ~0.35% fraud rate (comparable order of magnitude to real card-fraud
    data) with fraud concentrated at night, online, high distance-from-home,
    and price-ratio outliers.

Run:
    python src/generate_data.py --n 60000 --seed 42 --out data/transactions.csv
"""

import argparse
import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "dining", "fuel",
    "online_retail", "entertainment", "utilities", "jewelry", "cash_advance",
]
CHANNELS = ["card_present", "online", "phone_order"]


def generate(n: int, seed: int = 42, fraud_rate_target: float = 0.0035) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Latent "is this transaction actually fraud" ground truth, generated
    # first so downstream features can be shifted for fraud cases (mirrors
    # how real fraud differs systematically from legit spend).
    is_fraud = (rng.uniform(0, 1, n) < fraud_rate_target).astype(int)
    n_fraud = is_fraud.sum()

    hour = rng.integers(0, 24, n)
    # Fraud skews toward late night / early morning (mild, overlapping skew)
    hour[is_fraud == 1] = rng.choice(
        np.arange(24), n_fraud, p=_night_weighted_probs()
    )

    amount = np.round(np.exp(rng.normal(3.2, 1.1, n)), 2)  # log-normal spend
    amount[is_fraud == 1] = np.round(np.exp(rng.normal(3.7, 1.3, n_fraud)), 2)
    amount = np.clip(amount, 1.0, 25000.0)

    channel = rng.choice(CHANNELS, n, p=[0.55, 0.35, 0.10])
    channel[is_fraud == 1] = rng.choice(
        CHANNELS, n_fraud, p=[0.35, 0.55, 0.10]
    )
    card_present = (channel == "card_present").astype(int)

    merchant_category = rng.choice(MERCHANT_CATEGORIES, n)
    merchant_category[is_fraud == 1] = rng.choice(
        MERCHANT_CATEGORIES, n_fraud,
        p=[0.05, 0.15, 0.09, 0.05, 0.04, 0.22, 0.08, 0.03, 0.16, 0.13],
    )

    distance_from_home_km = np.clip(rng.exponential(8, n), 0, 3000)
    distance_from_home_km[is_fraud == 1] = np.clip(
        rng.exponential(60, n_fraud), 0, 5000
    )

    distance_from_last_txn_km = np.clip(rng.exponential(4, n), 0, 2000)
    distance_from_last_txn_km[is_fraud == 1] = np.clip(
        rng.exponential(45, n_fraud), 0, 4000
    )

    ratio_to_median_price = np.clip(rng.lognormal(0, 0.4, n), 0.05, 20)
    ratio_to_median_price[is_fraud == 1] = np.clip(
        rng.lognormal(0.7, 0.6, n_fraud), 0.1, 40
    )

    is_foreign = rng.choice([0, 1], n, p=[0.93, 0.07])
    is_foreign[is_fraud == 1] = rng.choice([0, 1], n_fraud, p=[0.70, 0.30])

    # 10 anonymized "PCA-style" continuous features. V1-V4 carry a real but
    # noisy, partially-overlapping fraud signal; V5-V10 are closer to pure
    # noise -- similar in spirit to real anonymized datasets where only a
    # subset of components separate the classes, and imperfectly at that.
    V = rng.normal(0, 1, size=(n, 10))
    fraud_shift = np.array([-1.1, 1.0, -0.85, 0.75, 0.15, -0.1, 0.05, 0.2, -0.15, 0.1])
    # Only ~70% of fraud cases actually carry the shifted signal, the rest
    # look statistically close to legitimate transactions (harder cases).
    signal_mask = rng.uniform(0, 1, n_fraud) < 0.70
    shift_applied = np.outer(signal_mask, fraud_shift)
    V[is_fraud == 1] += shift_applied + rng.normal(0, 1.0, size=(n_fraud, 10))

    df = pd.DataFrame({f"V{i+1}": V[:, i].round(4) for i in range(10)})
    df.insert(0, "transaction_id", np.arange(1, n + 1))
    df["hour"] = hour
    df["amount"] = amount
    df["channel"] = channel
    df["card_present"] = card_present
    df["merchant_category"] = merchant_category
    df["distance_from_home_km"] = distance_from_home_km.round(2)
    df["distance_from_last_txn_km"] = distance_from_last_txn_km.round(2)
    df["ratio_to_median_price"] = ratio_to_median_price.round(3)
    df["is_foreign_transaction"] = is_foreign
    df["is_fraud"] = is_fraud

    # Small amount of realistic missingness + duplicate rows for the
    # preprocessing step to handle.
    for col in ["distance_from_last_txn_km", "merchant_category"]:
        mask = rng.uniform(0, 1, n) < 0.01
        df.loc[mask, col] = np.nan
    dup_idx = rng.choice(df.index, size=max(1, n // 500), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _night_weighted_probs():
    # Weight hours 0-5 much more heavily for fraud
    w = np.ones(24)
    w[0:6] = 6.0
    w[22:24] = 3.0
    return w / w.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/transactions.csv")
    args = ap.parse_args()

    data = generate(args.n, args.seed)
    data.to_csv(args.out, index=False)
    print(f"Wrote {len(data)} rows to {args.out}")
    print(f"Fraud rate: {data['is_fraud'].mean():.5f} ({data['is_fraud'].sum()} fraud rows)")
