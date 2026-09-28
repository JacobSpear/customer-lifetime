import json
import sqlite3
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from csurv.features import build_features
from pathlib import Path


def load_churn_truth(churn_path: str = "data/churn.json") -> pd.DataFrame:
    with open(churn_path) as f:
        raw = json.load(f)
    df = pd.DataFrame([
        {"visit_id": int(k), "churned": v} for k, v in raw.items()
    ])
    return df


def load_model(model_path: str = "models/demo.json") -> tuple[dict, dict, list]:
    with open(model_path) as f:
        raw = json.load(f)
    betas = {k: np.array(v) for k, v in raw["betas"].items()}
    mean_sd = raw["mean_sd"]
    covariates = list(mean_sd.keys())
    return betas, mean_sd, covariates


def churn_summary(db_path: str = "data/demo.db",
                  churn_path: str = "data/churn.json",
                  model_path: str = "models/demo.json"):

    # build the same feature matrix the model saw
    df = build_features(db_path)
    truth = load_churn_truth(churn_path)
    df = df.merge(truth, on="visit_id", how="left")

    betas, mean_sd, covariates = load_model(model_path)

    # normalize the same way model.py does
    for cov in covariates:
        mu, sd = mean_sd[cov]
        df[cov] = (df[cov] - mu) / sd

    df[covariates] = df[covariates].fillna(0)

    # compute predicted pi
    ones = np.ones((len(df), 1))
    features = np.hstack((ones, df[covariates].values))
    pi_pred = sigmoid(features @ betas['pi'])
    df['pi_pred'] = pi_pred

    true_rate = df['churned'].mean()
    pred_rate = df['pi_pred'].mean()
    print(f"Overall churn rate   true: {true_rate:.3f}   predicted mean: {pred_rate:.3f}")
    print()

    # per-covariate comparison
    print("=" * 70)
    print(f"{'Covariate':<40s} {'beta_pi':>8s} {'corr w/ churn':>13s}")
    print("=" * 70)
    for i, cov in enumerate(covariates):
        beta_val = betas['pi'][i + 1]
        corr = df[[cov, 'churned']].corr().iloc[0, 1]
        match = "OK" if (beta_val > 0) == (corr > 0) else "MISMATCH"
        if abs(corr) < 0.005:
            match = "~0"
        print(f"{cov:<40s} {beta_val:>+8.3f} {corr:>+13.3f}   {match}")

    print()

    # subpopulation breakdowns using raw (un-normalized) columns
    conn = sqlite3.connect(db_path)
    visits_raw = pd.read_sql("SELECT visit_id, reservation_type, special_offer_category FROM VISITS", conn)
    customers_raw = pd.read_sql("SELECT * FROM CUSTOMERS", conn)
    feats_raw = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()

    feats_wide = feats_raw.pivot(index="customer_id", columns="feature_name", values="feature_value")
    raw = (visits_raw
           .merge(truth, on="visit_id", how="left")
           .merge(pd.read_sql("SELECT visit_id, customer_id FROM VISITS",
                              sqlite3.connect(db_path)),
                  on="visit_id", how="left")
           .merge(feats_wide, on="customer_id", how="left"))

    for col in ["reservation_type", "VIP", "first_reservation_source", "special_offer_category"]:
        if col in raw.columns:
            print(f"=== Churn by {col} ===")
            grp = raw.groupby(col)["churned"].agg(["mean", "count"])
            grp.columns = ["churn_rate", "n"]
            print(grp.sort_values("churn_rate"))
            print()


if __name__ == "__main__":
    churn_summary()
