import numpy as np, pandas as pd

from vivarium import Artifact
art = Artifact('src/vivarium_uw_covid/artifacts/new_york.hdf')
df_covs = art.load('beta.covariates')
coeffs = art.load('beta.coeffs')
compartment_sizes = art.load('seiir.compartment_sizes')

t0 = '2020-09-01'
initial_states = compartment_sizes.loc[t0].set_index('draw').dropna()
params = art.load('seiir.params')


from vivarium_uw_covid.components.seiir_beta import *
from vivarium_uw_covid.components.seiir_agent import *



def test_run_agent_model():
    beta = beta_predict(coeffs, df_covs)

    n_draws = 1
    df_list = run_agent_model(n_draws, n_simulants=100_000,
                                      params=params, beta=beta,
                                      start_time=t0, initial_states=initial_states)

    assert len(df_list) == n_draws
    for state in ['S', 'E', 'I1', 'I2', 'R', 'n_new_infections', 'n_new_isolations']:
        assert state in df_list[0].columns, f'expect column for state "{state}"'

