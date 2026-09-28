from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    db_path: str
    model_path: str = "models/demo.json"
    horizon_days: int = 90
    avg_ticket: float = 65.0
    churn_reduction_factor: float = 1.0

class CustomerPrediction(BaseModel):
    visit_id: int
    pi: float
    k: float
    lam: float
    mean_return: float
    median_return: float
    expected_visits: float
    clv: float | None

class PredictResponse(BaseModel):
    predictions: list[CustomerPrediction]
    n_customers: int
    mean_churn: float
    mean_clv: float | None

class UpliftResponse(BaseModel):
    mean_clv_baseline: float
    mean_clv_reduced: float
    mean_clv_uplift: float
    mean_clv_uplift_pct: float


class SurvivalPoint(BaseModel):
    day: float
    survival_probability: float


class SurvivalResponse(BaseModel):
    curve: list[SurvivalPoint]
    n_customers: int
    cure_fraction: float = Field(description="Mean 1-pi (long-run survival floor)")