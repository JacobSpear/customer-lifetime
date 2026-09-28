from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from csurv.predict import (
    load_model, prepare_features, compute_parameters,
    predict_customer, predict_clv_uplift
)
from csurv.features import build_features
from api.schemas import *
from scipy.special import gamma
import numpy as np

MODEL = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL['betas'], MODEL['mean_sd'], MODEL['covariates'] = load_model("models/demo.json")
    yield
    MODEL.clear()


app = FastAPI(title="csurv", version="0.1.0", lifespan=lifespan)


# ── Shared helper ───────────────────────────────────────────────
def _build_theta(db_path: str):
    """Build features → theta from a database. Returns (df, theta)."""
    betas, mean_sd, covariates = MODEL["betas"], MODEL["mean_sd"], MODEL["covariates"]
    df = build_features(db_path)
    df = df[df["event"] == 0].copy()
    features = prepare_features(df, mean_sd, covariates)
    theta = compute_parameters(betas, features)
    return df, theta


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok",
            "model_path": "models/demo.json",
            "n_covariates": len(MODEL["covariates"])}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    df, theta = _build_theta(req.db_path)
    results = predict_customer(theta, req.horizon_days, req.avg_ticket)
    results.index = df["visit_id"].values
    predictions = results.reset_index(names="visit_id").to_dict(orient="records")

    return PredictResponse(
        predictions=predictions,
        n_customers=len(results),
        mean_churn=float(results["pi"].mean()),
        mean_clv=float(results["clv"].mean()),
    )


@app.post("/uplift", response_model=UpliftResponse)
def uplift_endpoint(req: PredictRequest):
    if req.churn_reduction_factor >= 1.0:
        raise HTTPException(400, "churn_reduction_factor must be < 1.0 for uplift analysis")

    df, theta = _build_theta(req.db_path)
    up = predict_clv_uplift(theta, req.horizon_days, req.avg_ticket,
                            req.churn_reduction_factor)

    return UpliftResponse(
        mean_clv_baseline=float(up["clv_baseline"].mean()),
        mean_clv_reduced=float(up["clv_reduced"].mean()),
        mean_clv_uplift=float(up["clv_chg"].mean()),
        mean_clv_uplift_pct=float(up["clv_pctchg"].mean()),
    )


@app.post("/survival", response_model=SurvivalResponse)
def survival_endpoint(req: PredictRequest):
    df, theta = _build_theta(req.db_path)
    pi, k, lam = theta["pi"], theta["k"], theta["lam"]

    t = np.linspace(0, req.horizon_days, 200)
    S = (1 - pi[np.newaxis, :]) * np.exp(
        -(t[:, np.newaxis] / lam[np.newaxis, :]) ** k[np.newaxis, :]
    )
    S_mean = S.mean(axis=1)

    curve = [SurvivalPoint(day=float(t[i]), survival_probability=float(S_mean[i]))
             for i in range(len(t))]

    return SurvivalResponse(
        curve=curve,
        n_customers=len(pi),
        cure_fraction=float((1 - pi).mean()),
    )