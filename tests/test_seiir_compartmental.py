import numpy as np, pandas as pd
from vivarium import Artifact
from vivarium_uw_covid.components.seiir_beta import *
from vivarium_uw_covid.components.seiir_compartmental import *


art = Artifact('src/vivarium_uw_covid/artifacts/new_york.hdf')
df_covs = art.load('beta.covariates')
coeffs = art.load('beta.coeffs')

params = art.load('seiir.params')
compartment_sizes = art.load('seiir.compartment_sizes')
t0 = '2020-09-01'
t1 = '2020-09-05'
initial_states = compartment_sizes.loc[t0].set_index('draw').dropna()


def test_run_compartmental_model():
    beta = beta_predict(coeffs, df_covs)

    n_draws = 2
    df_draws = run_compartmental_model(n_draws, n_simulants=100_000,
                                      params=params, beta=beta,
                                      start_time=t0, end_time=t1,
                                      initial_states=initial_states)

    assert len(df_draws) == n_draws
    keys = df_draws.keys()
    key = list(keys)[0]
    for state in ['S', 'E', 'I1', 'I2', 'R']:
        assert state in df_draws[key].columns, f'expect column for state "{state}"'

