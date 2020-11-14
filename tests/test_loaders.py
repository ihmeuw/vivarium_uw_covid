import numpy as np, pandas as pd

cov_dir = 'best' # most recent dir in $seiir_dir/covariate/
run_dir = 'best'  # updated, since old dirs were deleted
loc_id = 60886 # King and Snohomish Counties
t0 = pd.Timestamp('2020-09-01')

from vivarium_uw_covid.data.loader import *

def test_load_seiir_params():
    params = load_seiir_params(run_dir, theta=0)
    assert len(params.columns) == 1_000, 'expect 1,000 draws'
    for p in ['alpha', 'sigma', 'gamma1', 'gamma2']:
        assert p in params.index, f'expect row for "{p}" in DataFrame'


def test_load_beta_fit():
    load_beta_fit(run_dir, loc_id)


def test_load_uw_fac_staff_ages():
    load_uw_fac_staff_ages()


def test_load_ifr():
    load_ifr(run_dir)


def test_load_covariates():
    df_covs = load_covariates(cov_dir, loc_id)

    assert 'pneumonia' in df_covs.columns


def test_load_effect_coefficients():
    coeffs = load_effect_coefficients(run_dir, loc_id)
    assert 'intercept' in coeffs.columns, 'expect a mask_use effect coefficient'
    assert len(coeffs) == 1_000, 'expect 1,000 draws'


def test_load_seiir_compartment_sizes():
    df = load_seiir_compartment_sizes(run_dir, loc_id)
    for state in ['S', 'E', 'I1', 'I2', 'R']:
        assert state in df.columns, f'expect column for state "{state}"'
    assert df.draw.nunique() == 1_000, 'expect 1,000 draws'




