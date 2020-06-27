import numpy as np, pandas as pd


cov_dir = '2020_06_23.03.01' # most recent dir in $seiir_dir/covariate/
run_dir = '2020_06_23.07'  # 6/23 Production Runs, part 2, reference scenario
loc_id = 60886 # King and Snohomish Counties
t0 = pd.Timestamp('2020-09-01')


from vivarium_uw_covid.data.loader import *
from vivarium_uw_covid.components.seiir_beta import *


df_covs = load_covariates(cov_dir)
coeffs = load_effect_coefficients(run_dir, loc_id)
initial_states = load_seiir_initial_states(run_dir, loc_id, t0)
params = load_seiir_params(run_dir, theta=0)


def test_beta_predict():
    beta_pred = beta_predict(coeffs, df_covs)


def test_make_alternative_covariates():
    df_alt = make_alternative_covariates(df_covs, loc_id, t0, mask_use=.99)
    assert not np.allclose(df_alt.mask_use, .99)
    assert np.allclose(df_alt.mask_use.loc[t0:], .99)

