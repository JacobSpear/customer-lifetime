import pytest
import numpy as np
import pandas as pd
from csurv.predict import compute_parameters as cp, simulate_expected_visits as sev, prepare_features as pf



def test_sev_certainty():
    n_obs = 2
    pi = np.array([0.1, 1.0])  
    k = np.array([100.0, 100.0]) 
    lam = np.array([1.0, 1.0])
    visits = sev(pi, k, lam, horizon_days=10, n_sims=500)
    assert visits[1]==0 #boundary  
    assert visits.shape == (n_obs,) #contract
    assert 0 <= float(visits[0]) < 10 #invariant


def test_pf():

    df = pd.DataFrame({'labels':['a','b','c','d'],'col1':[0,1,2,5],'col2':[5,5,5,5]})
    covariates = ['col1','col2']
    mean_sd = {'col1':(2,2),'col2':(5,2)}
    n_obs=4
    features = pf(df,mean_sd,covariates)
    assert features.shape == (n_obs,3) #contract
    assert features[:,0].sum() == n_obs #invariant
    np.testing.assert_array_almost_equal(features[:,2] , [0,0,0,0]) #boudary
    assert features[0,1] < 0 
    assert features[3,1] > 0

