from scipy.special import expit as sigmoid
from scipy.special import log_expit as log_sigmoid
from scipy.stats import weibull_min as weibull
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from csurv.features import build_features
import json



def normalize(df:pd.DataFrame,covariates:list) -> tuple[pd.DataFrame,dict]:
    """Takes in a dataframe and a feature (column) name.  Returns a dataframe with the same
    column names, but that feature is linearly scaled to have mean 0 and variance 1.
    Returns the original means and SDs in a dict with key value pairs "feature name":(mean,sd)""" 
    dist_dict={}
    for feature in covariates:
        mu,sd = df[feature].mean(),df[feature].std()
        df[feature] = (df[feature]-mu)/sd
        dist_dict[feature]=(mu,sd)
    return df,dist_dict

def compute_parameters(betas:dict,features:np.ndarray) -> dict:
    theta = {}
    theta['pi'] = sigmoid(features @ betas['pi'])
    theta['lam'] = np.exp(np.clip(features @ betas['lam'], -3, 5))
    theta['k'] = np.exp(np.clip(features @ betas['k'], -3, 3))
    return theta

def E_step(betas:dict,times:np.ndarray,events:np.ndarray,features:np.ndarray,n_obs:int) -> np.ndarray:
    """Returns a np.ndarray containing a probability of churn for each observation.
    The return probability should be 0 for uncensored rows (events=1) and should be 
    theta['pi']/(theta['pi']+(1-theta['pi'])(1-WeibullCDF(times,theta['k'],theta['lam']))
    for censored rows.
    """
    theta = compute_parameters(betas, features)
    pi,k,lam = theta['pi'],theta['k'],theta['lam']
    latency = (weibull.sf(times,k,scale=lam))
    P_tilde = np.where(events == 1, 0, 
                       (pi
                        /
                        (pi+ (1-pi)*latency)
                        )
    )
    
    return(P_tilde)
    

def M_step(P_tilde:np.ndarray,features:np.ndarray,betas:dict,times:np.ndarray,events:np.ndarray) -> tuple[dict,float]:
    """
    P_tilde is an array of shape (n_obs,) which is the output of E_step and contains the probability that each
    observation represents churn given the data and the previously observed theta.

    The output is a new set of betas such that the Expectation of log likelihood is increased
    """
    n = features.shape[1]
    #The expectation of log-likelihood decomposes as 
    # H_w(P_tilde,sigmoid(x @ beta_pi)) (weighted cross entropy)
    # + H_w(1-P_tilde,1-weibull_pdf(times,beta_k,beta_lambda))
    # Because the first term depends only on beta_pi and the second
    # depends on beta_k and beta_lambda we can optimize each separately
    def incidence_ll(beta_pi:np.ndarray,P_tilde:np.ndarray, features:np.ndarray) -> float:
        pre_pi = features @ beta_pi
        return -1 * ((1-P_tilde) @ log_sigmoid(-pre_pi) + P_tilde @ log_sigmoid(pre_pi))

    def latency_ll(betavec, P_tilde, features, times, events):
        beta_lam = betavec[:features.shape[1]]
        beta_k   = betavec[features.shape[1]:]
        k   = np.exp(np.clip(features @ beta_k,  -3, 3))
        lam = np.exp(np.clip(features @ beta_lam, -3, 5))

        uncensored = (events == 1)
        censored   = ~uncensored

        ll  = (1 - P_tilde[uncensored]) @ weibull.logpdf(times[uncensored], k[uncensored], scale=lam[uncensored])
        ll += (1 - P_tilde[censored])   @ weibull.logsf(times[censored], k[censored], scale=lam[censored])

        return -ll
    
    def jac_pi(beta_pi:np.ndarray,
               P_tilde:np.ndarray,
               features:np.ndarray) -> np.ndarray:
        pi = sigmoid(features @ beta_pi)
        return features.T @ (pi - P_tilde)
    bounds_pi = [(-1, 1)] * n  # pi bounds
    
    beta_pi_out = minimize(incidence_ll,
                           x0=betas['pi'],
                           args=(P_tilde,features),
                           method='L-BFGS-B',
                           jac=jac_pi
                           )
    print(f"  incidence converged: {beta_pi_out.success}, fun: {beta_pi_out.fun}")

    betas_vec = np.concatenate([betas['lam'],betas['k']])
    
    bounds = [(-1, 1)] * n + [(-1, 1)] * n  # k bounds, then lam bounds

    beta_vec_out = minimize(latency_ll,
                           x0=betas_vec,
                           args=(P_tilde,features,times,events),
                           method='L-BFGS-B')
    print(f"  latency converged: {beta_vec_out.success}, fun: {beta_vec_out.fun}")
    print(f"  latency message: {beta_vec_out.message}")

    beta_k_out = beta_vec_out.x[n:]
    beta_lam_out = beta_vec_out.x[:n]
    new_betas= {}
    new_betas['pi']=beta_pi_out.x
    new_betas['lam']=beta_lam_out
    new_betas['k']=beta_k_out
    theta = compute_parameters(new_betas, features)
    pi,k,lam = theta['pi'],theta['k'],theta['lam']

    uncensored = events == 1
    censored = ~uncensored

    ll_uncensored = np.sum(np.log(1 - pi[uncensored]) 
                        + weibull.logpdf(times[uncensored], k[uncensored], scale=lam[uncensored]))
    #print(f"pi:{np.isnan(pi).sum()}   k:{np.isnan(k).sum()}  lam:{np.isnan(lam).sum()}")
    ll_censored = np.sum(np.log(pi[censored] 
                        + (1 - pi[censored]) * weibull.sf(times[censored], k[censored], scale=lam[censored])))
    ll_out =  -(beta_pi_out.fun + beta_vec_out.fun)

    return new_betas,ll_out

def EMCure_Learn(dbpath : str ="data/demo.db",
                 time:str = 'time',
                 event:str='event',covariates:list =[],
                 seed:int=42,epsilon:float = 0.00001) -> tuple[dict,dict]:
    df = build_features(dbpath)
    
    if len(covariates)==0:
        exclude = ['time', 'event', 'customer_id', 'visit_id', 'visit_date',
           'first_visit_date', 'gap_num', 'days_since_first_visit','entree','avg_gap_days','prev_gap_days','VIP_None']
        covariates = [c for c in df.columns if c not in exclude 
                    and pd.api.types.is_numeric_dtype(df[c])]
    
    n_obs = len(df)
    onehot_groups = {}
    for col in covariates:
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[0] in ['VIP', 'reservation_type', 'first_reservation_source',
                                            'Amex Member', 'special_offer_category']:
            onehot_groups.setdefault(parts[0], []).append(col)

    drop_cols = [cols[0] for cols in onehot_groups.values()]
    covariates = [c for c in covariates if c not in drop_cols]
    print(f"Remaining Covariates: {covariates}")
    df[covariates]=df[covariates].fillna(0)
    n_features = len(covariates)
    if n_features > 0:
        df , mean_sd = normalize(df,covariates)
    else:
        df , mean_sd = df , {}

    ones = np.ones((n_obs, 1))
    features = np.hstack((ones, df[covariates].values))
    times = df[time].values.astype(float)
    #time=0 breaks the weibull logpdf
    times = np.maximum(times, 0.5)
    events = df[event].values.astype(float)
    rng = np.random.default_rng(seed)
    #initialize betas
    betas={}
    betas['pi'] = rng.uniform(-2,2,n_features+1)
    betas['lam']= rng.uniform(-0.02,0.02,n_features+1)
    betas['k'] = rng.uniform(-0.02,0.02,n_features+1)

    old_LL = np.inf
    new_LL = -np.inf
    beta_delta = np.inf
    i =0
    while beta_delta>epsilon or i<3:
        #temp_LL=new_LL
        old_betas = {k: v.copy() for k, v in betas.items()}
        P_tilde = E_step(betas, times, events, features, n_obs)
        betas, new_LL = M_step(P_tilde, features, betas, times, events)

        beta_delta = max(
            np.max(np.abs(betas[k] - old_betas[k])) for k in betas
        )

        print(f"Iter {i}: LL = {new_LL}, max |Δβ| = {beta_delta}")
        #P_tilde = E_step(betas,times,events,features,n_obs)
        #betas, new_LL = M_step(P_tilde,features, betas,times,events)
        #old_LL=temp_LL
        i+=1
        #print(f"Iter {i}: LL = {new_LL}")
    return betas , mean_sd




