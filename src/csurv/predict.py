from scipy.special import expit as sigmoid
from scipy.stats import weibull_min as weibull
from scipy.special import gamma
import numpy as np
import pandas as pd
import json
from csurv.features import build_features
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


def load_model(model_path: str = "models/demo.json") -> tuple[dict, dict, list]:
    with open(model_path) as f:
        model_dict = json.load(f)
    betas = {k: np.array(v) for k, v in model_dict["betas"].items()}
    mean_sd = model_dict["mean_sd"]
    covariates = list(mean_sd.keys())
    return betas, mean_sd, covariates


def prepare_features(df: pd.DataFrame, mean_sd: dict, covariates: list) -> np.ndarray:
    """Normalize covariates using saved mean/sd, prepend intercept column.
    Returns feature matrix of shape (n_obs, n_covariates + 1)."""
    n_obs = len(df)
    df2 = df.copy()
    for feature in covariates:
        df2[feature]=(df2[feature] - mean_sd[feature][0])/mean_sd[feature][1]
    df2[covariates] = df2[covariates].fillna(0)
    ones = np.ones((n_obs, 1))
    features = np.hstack((ones, df2[covariates].values))
    return features


def compute_parameters(betas:dict,features:np.ndarray) -> dict:
    theta = {}
    theta['pi'] = sigmoid(features @ betas['pi'])
    theta['lam'] = np.exp(np.clip(features @ betas['lam'], -3, 5))
    theta['k'] = np.exp(np.clip(features @ betas['k'], -3, 3))
    return theta

def simulate_expected_visits(pi: np.ndarray, k: np.ndarray, lam: np.ndarray,
                             horizon_days: float = 90,
                             n_sims: int = 1000,
                             seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    n_obs = len(pi)

    pi_mat = np.broadcast_to(pi, (n_sims, n_obs))
    k_mat = np.broadcast_to(k, (n_sims, n_obs))
    lam_mat = np.broadcast_to(lam, (n_sims, n_obs))

    total_visits = np.zeros((n_sims, n_obs))
    elapsed = np.zeros((n_sims, n_obs))
    active = np.ones((n_sims, n_obs), dtype=bool)

    max_visits = int(horizon_days)  # upper bound to prevent infinite loop

    for _ in range(max_visits):
        # Check who is still active
        if not active.any():
            break

        # Did the customer churn?
        churned = rng.random((n_sims, n_obs)) < pi_mat
        active &= ~churned

        # If not, how long until they return? Did they return "in time"?
        gap = rng.weibull(k_mat) * lam_mat  # Weibull(k, scale=lam)
        elapsed += gap * active
        within_horizon = elapsed <= horizon_days
        arrived = active & within_horizon

        total_visits += arrived

        # Deactivate those who exceeded horizon
        active &= within_horizon
        return total_visits.mean(axis=0)


def predict_customer(theta: dict, horizon_days: float = 90,
                     avg_ticket: float | None = 65) -> pd.DataFrame:
    """Compute per-observation predictions from theta.

    Returns DataFrame with columns:
        pi              - churn probability
        k               - Weibull shape
        lam             - Weibull scale (days)
        mean_return     - expected return time (days) for susceptible customers
        median_return   - median return time (days) for susceptible customers
        expected_visits - expected visits in horizon (considering churn)
        clv             - expected_visits * avg_ticket (None if avg_ticket not given)
    """
    pi = theta['pi']
    k = theta['k']
    lam = theta['lam']
    mean_return = lam * gamma(1+1/k)
    median_return = lam * (np.log(2)) ** (1/k)
    expected_visits = simulate_expected_visits(theta['pi'], theta['k'], theta['lam'],
                                           horizon_days=horizon_days)
    clv = expected_visits * avg_ticket

    return pd.DataFrame({'pi':pi,'k':k,'lam':lam,
                         'mean_return':mean_return,
                         'median_return':median_return,
                         'expected_visits':expected_visits,
                         'clv':clv})

def predict_clv_uplift(theta: dict, horizon_days: float = 90,
                       avg_ticket: float = 1.0,
                       churn_reduction_factor: float = 0.90) -> pd.DataFrame:

    theta2 = theta.copy()
    theta2['pi'] = theta['pi']*churn_reduction_factor
    clv_baseline = predict_customer(theta, horizon_days, avg_ticket)['clv']
    clv_reduced = predict_customer(theta2,horizon_days,avg_ticket)['clv']
    clv_chg = clv_reduced-clv_baseline
    with np.errstate(divide='ignore'):
        clv_pctchg = np.divide( clv_chg , clv_baseline)

    return pd.DataFrame({'clv_baseline': clv_baseline,'clv_reduced':clv_reduced,'clv_chg': clv_chg,
                         'clv_pctchg': clv_pctchg})
    



def marginal_clv(df: pd.DataFrame, betas: dict, mean_sd: dict,
                 covariates: list, covariate: str,
                 horizon_days: float = 90,
                 avg_ticket: float = 1.0) -> dict:
    """Average Marginal Effect of a covariate on CLV.

    For binary covariates: compute CLV for all obs with covariate=1 vs =0,
        return mean difference.
    For continuous covariates: compute CLV at +1sd vs -1sd,
        return mean difference.

    Returns dict with keys:
        covariate, clv_high, clv_low, marginal_clv, is_binary
    """
    dfs=[df.copy(), df.copy()]

    out_dict = {'covariate':covariate}
    trans = ['clv_low','clv_high']
    for i, _df in enumerate(dfs):
        if set(df[covariate].unique()).issubset({np.nan,0,1}):
            is_binary = 1
            _df[covariate] = i
        else:
            is_binary=0
            _df[covariate] = df[covariate] + (2*i-1) * mean_sd[covariate][1]
        
        out_dict['is_binary']=is_binary
        _df2 = prepare_features(_df,mean_sd,covariates)
        out_dict[trans[i]] = predict_customer(compute_parameters(betas,_df2),
                              horizon_days, avg_ticket)['clv'].mean()
    out_dict['marginal_clv'] = out_dict['clv_high'] - out_dict['clv_low']
        

        
    return out_dict

def marginal_clv_table(df, betas, mean_sd, covariates,
                       horizon_days=90, avg_ticket=1.0) -> pd.DataFrame:
    # This is just: map marginal_clv over covariates, collect into DataFrame
    rows = [marginal_clv(df, betas, mean_sd, covariates, cov,
                         horizon_days, avg_ticket)
            for cov in covariates]
    return pd.DataFrame(rows).sort_values("marginal_clv", ascending=False)


def simulate_churn_times(pi, k, lam, horizon_days=90,
                        n_sims=1000, seed=42):
    """
    Uses estimated parameters to predict time to simulate churn dynamics.
    """
    rng = np.random.default_rng(seed)
    n_obs = len(pi)

    pi_mat  = np.broadcast_to(pi,  (n_sims, n_obs))
    k_mat   = np.broadcast_to(k,   (n_sims, n_obs))
    lam_mat = np.broadcast_to(lam, (n_sims, n_obs))

    elapsed = np.zeros((n_sims, n_obs))
    event_observed = np.zeros((n_sims, n_obs), dtype=bool)
    active = np.ones((n_sims, n_obs), dtype=bool)

    max_iters = int(horizon_days)

    for _ in range(max_iters):
        if not active.any():
            break

        churned = rng.random((n_sims, n_obs)) < pi_mat
        just_churned = active & churned
        event_observed |= just_churned
        active &= ~churned

        gap = rng.weibull(k_mat) * lam_mat
        elapsed += gap * active

        active &= (elapsed <= horizon_days)

    # restrict elapsed for simulations that ran past
    elapsed = np.minimum(elapsed, horizon_days)

    # get mean duration and churn probability
    durations = elapsed.mean(axis=0)                  # shape (n_obs,)
    churn_rate = event_observed.mean(axis=0)           # shape (n_obs,)

    # KM needs a binary event flag per customer.
    # Threshold: if customer churned in >50% of sims, mark as observed.
    observed=rng.binomial(1, churn_rate).astype(bool)

    return durations, observed

def survival_plot(times, theta, horizon_days=90, max_last_visit_factor=5):
    """KM survival curve from simulated churn times."""


    pi, k, lam = theta['pi'], theta['k'], theta['lam']

    # Filter: exclude customers whose expected return is so long
    # they're probably already gone
    mean_return = lam * gamma(1 + 1/k)
    mask = times < max_last_visit_factor * mean_return.mean()
   
    durations, observed = simulate_churn_times(
        pi[mask], k[mask], lam[mask], horizon_days
    )

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=observed,
            timeline=np.linspace(0, horizon_days, 200),
            label='Simulated customer survival')

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Days from now')
    ax.set_ylabel('P(still active)')
    ax.set_title('Customer Survival — Simulated KM')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('survival_curve.png', dpi=150)
    plt.show()
    print("Saved: survival_curve.png")
    
def predict(db_path, model_path="models/demo.json",
            horizon_days=90, avg_ticket=65, churn_reduction_factor=0.90) -> pd.DataFrame:

    betas, mean_sd, covariates = load_model(model_path)
    df = build_features(db_path)
    df = df[df['event'] == 0].copy()

    # Save raw data before prepare_features mutates df
    times = df['time'].values.copy()

    features = prepare_features(df, mean_sd, covariates)
    theta = compute_parameters(betas, features)

    # Core predictions
    results = predict_customer(theta, horizon_days, avg_ticket)
    results.index = df['visit_id'].values

    # CLV uplift
    if churn_reduction_factor > 0 and avg_ticket is not None:
        uplift = predict_clv_uplift(theta, horizon_days, avg_ticket, churn_reduction_factor)
        results = pd.concat([results, uplift], axis='columns')

    # Survival plot — pass raw times for the filter
    survival_plot(times, theta, horizon_days)

    # Marginal CLV table — pass raw df so prepare_features normalizes correctly
    if avg_ticket is not None:
        ame = marginal_clv_table(df, betas, mean_sd, covariates,
                                 horizon_days, avg_ticket)
        print("\nMarginal CLV (Average Marginal Effects):")
        print(ame.to_string(index=False))

    return results



    


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo.db"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/demo.json"

    results = predict(db_path, model_path, horizon_days=90, avg_ticket=85.0)
    print(results.describe())
    print()
    print("Top 10 highest churn risk:")
    print(results.nlargest(10, "pi")[["pi", "mean_return", "clv"]])
