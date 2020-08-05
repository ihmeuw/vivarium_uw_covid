import numpy as np, pandas as pd
from vivarium import Artifact
from vivarium_uw_covid.components.seiir_beta import *
from vivarium_uw_covid.components.seiir_hybrid import *


art = Artifact('src/vivarium_uw_covid/artifacts/new_york.hdf')
df_covs = art.load('beta.covariates')
coeffs = art.load('beta.coeffs')
params = art.load('seiir.params')
compartment_sizes = art.load('seiir.compartment_sizes')
t0 = '2020-09-01'
t1 = '2020-09-05'
initial_states = compartment_sizes.loc[t0].set_index('draw').dropna()
beta = beta_predict(coeffs, df_covs)


def test_run_hybrid_model():

    n_draws = 1
    df_list_1, df_list_2 = run_hybrid_model(n_draws, n_simulants=100_000,
                                            mixing_parameter=.5, params=params,
                                            beta_agent=beta, beta_compartment=beta,
                                            start_time=t0,
                                            end_time=t1,
                                            initial_states_agent=initial_states,
                                            initial_states_compartment=initial_states)

    assert len(df_list_1) == n_draws
    for state in ['S', 'E', 'I1', 'I2', 'R', 'n_new_infections', 'n_new_isolations']:
        assert state in df_list_1[0].columns, f'expect column for state "{state}"'

def test_run_hybrid_model_w_testing():

    n_draws = 1
    df_list_1, df_list_2 = run_hybrid_model(n_draws, n_simulants=100_000,
                                            mixing_parameter=.5, params=params,
                                            beta_agent=beta, beta_compartment=beta,
                                            start_time=t0,
                                            end_time=t1,
                                            initial_states_agent=initial_states,
                                            initial_states_compartment=initial_states,
                                            use_mechanistic_testing=True,
                                            test_rate=0.001,
                                            test_positive_rate=0.05)

    assert len(df_list_1) == n_draws
    for state in ['S', 'E', 'I1', 'I2', 'R', 'n_new_infections', 'n_new_isolations']:
        assert state in df_list_1[0].columns, f'expect column for state "{state}"'

