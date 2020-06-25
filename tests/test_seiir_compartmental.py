import numpy as np, pandas as pd


cov_dir = '2020_06_23.03.01' # most recent dir in $seiir_dir/covariate/
run_dir = '2020_06_23.07'  # 6/23 Production Runs, part 2, reference scenario
loc_id = 60886 # King and Snohomish Counties
t0 = pd.Timestamp('2020-09-01')


from vivarium_uw_covid.data.loader import *
from vivarium_uw_covid.components.seiir_beta import *
from vivarium_uw_covid.components.seiir_compartmental import *


df_covs = load_covariates(cov_dir)
coeffs = load_effect_coefficients(run_dir, loc_id)
initial_states = load_seiir_initial_states(run_dir, loc_id, t0)
params = load_seiir_params(run_dir, theta=0)


def test_run_compartmental_model():
    beta = beta_predict(coeffs, df_covs[df_covs.location_id == loc_id])

    n_draws = 1
    df_list = run_compartmental_model(n_draws, n_simulants=100_000,
                                      params=params, beta=beta,
                                      start_time=t0, initial_states=initial_states)

    assert len(df_list) == n_draws
    for state in ['S', 'E', 'I1', 'I2', 'R']:
        assert state in df_list[0].columns, f'expect column for state "{state}"'

