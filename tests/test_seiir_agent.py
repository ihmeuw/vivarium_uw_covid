import numpy as np, pandas as pd
from vivarium_uw_covid.components.seiir_agent import *

# instead of loading an artifact, make testing data directly
# based on 2020_08_04a_covid_microsim_seiir_tufts_params.ipynb
n_draws = 32
t0 = '2020-09-01'
t1 = '2020-09-05'

params = pd.DataFrame(
               {'alpha':[1.0]*n_draws,
                'sigma': [1/3.0]*n_draws,  # my sigma is one over their theta
                'gamma1': [0.0102]*n_draws, # my gamma1 is their sigma
                'gamma2': [1/14.0]*n_draws, # my gamma2 is their rho
                'theta': [5/7]*n_draws  # my theta is their X*7 I think) --- X is imported cases per week
               }, index=[str(x) for x in range(n_draws)]).T  # TODO: make this index cooler

initial_states = pd.DataFrame({'S': [4_990]*n_draws,
                               'E': [0]*n_draws,
                               'I1': [10]*n_draws,
                               'I2': [0]*n_draws,
                               'R':[0]*n_draws})

beta = pd.DataFrame(index=pd.date_range(t0, t1))
for draw in range(n_draws):
    beta[draw] = 0.085



def test_run_agent_model():
    n_draws = 2
    df_draws = run_agent_model(n_draws, n_simulants=100_000,
                              params=params, beta=beta,
                              start_time=t0, end_time=t1,
                              initial_states=initial_states)

    assert len(df_draws) == n_draws
    keys = df_draws.keys()
    key = list(keys)[0]
    for state in ['S', 'E', 'I1', 'I2', 'R', 'n_new_infections', 'n_new_isolations']:
        assert state in df_draws[key].columns, f'expect column for state "{state}"'

