import numpy as np, pandas as pd

cov_dir = '2020_06_23.03.01' # most recent dir in $seiir_dir/covariate/
run_dir = '2020_06_23.07'  # 6/23 Production Runs, part 2, reference scenario
loc_id = 60886 # King and Snohomish Counties
t0 = pd.Timestamp('2020-09-01')

from vivarium_uw_covid.data.loader import *

def test_load_covariates():
    df_covs = load_covariates(cov_dir)

    assert 'mask_use' in df_covs.columns


def test_load_coeffs():
    coeffs = load_effect_coefficients(run_dir, loc_id)
    assert 'mask_use' in coeffs.columns, 'expect a mask_use effect coefficient'
    assert len(coeffs) == 1_000, 'expect 1,000 draws'


def test_load_seiir_initial_states():
    initial_states = load_seiir_initial_states(run_dir, loc_id, t0)
    for state in ['S', 'E', 'I1', 'I2', 'R']:
        assert state in initial_states.columns, f'expect column for state "{state}"'
    assert len(initial_states) == 1_000, 'expect 1,000 draws'


def test_load_seiir_params():
    params = load_seiir_params(run_dir, theta=0)
    assert len(params) == 1_000, 'expect 1,000 draws'
    for p in ['alpha', 'sigma', 'gamma1', 'gamma2']:
        assert p in params[999].keys(), f'expect parameter "{p}" to be in dict'



