"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd


def agent_covid_initial_states(n_simulants, compartment_sizes):
    """Initialize indiviuals to COVID model states of S, E, I1, I2, R
    
    Parameters
    ----------
    
    n_simulants : int
    compartment_sizes : dict_like, with numbers keyed by S, E, I1, I2, R
    """
    state_list = ['S', 'E', 'I1', 'I2', 'R']
    
    p = [compartment_sizes[state] for state in state_list]
    p = np.array(p)
    p /= p.sum()
    
    state = np.random.choice(state_list, size=n_simulants, replace=True, p=p)
    return state


def agent_covid_step(df, alpha, beta, gamma1, gamma2, sigma, theta):
    # find the number infectious
    n_infectious = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants = len(df)
    uniform_random_draw = np.random.uniform(size=n_simulants)

    # update from R back to S, to allow in-place computation
    pr_I2_to_R = 1 - np.exp(-gamma2)
    rows = (df.covid_state == 'I2') & (uniform_random_draw < pr_I2_to_R)
    df.loc[rows, 'covid_state'] = 'R'

    pr_I1_to_I2 = 1 - np.exp(-gamma1)
    rows = (df.covid_state == 'I1') & (uniform_random_draw < pr_I1_to_I2)
    df.loc[rows, 'covid_state'] = 'I2'

    pr_E_to_I1 = 1 - np.exp(-sigma)
    rows = (df.covid_state == 'E') & (uniform_random_draw < pr_E_to_I1)
    df.loc[rows, 'covid_state'] = 'I1'

    pr_infected = 1 - np.exp(-(beta * n_infectious**alpha + theta)/ n_simulants)
    rows = (df.covid_state == 'S') & (uniform_random_draw < pr_infected)
    df.loc[rows, 'covid_state'] = 'E'
    return np.sum(rows) # new_infections


def run_agent_model(n_draws, n_simulants, params, beta, start_time, initial_states):
    """Project population sizes from start time to end of beta.index

    Parameters
    ----------
    n_draws : int
    n_simulants : int
    params : dict-of-dicts, where draw maps to dict with values for 
                alpha, gamma1, gamma2, sigma, theta : float
    beta : pd.DataFrame with index of dates and columns for draws
    start_time : pd.Timestamp
    initial_states : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R

    Results
    -------
    returns list of pd.DataFrames with colunms for counts for S, E, I1, I2, and R
    and rows for each day of projection
    """
    df_count_list = []

    for draw in np.random.choice(range(1_000), replace=False, size=n_draws):
        df = pd.DataFrame(index=range(n_simulants))
        df['covid_state'] = agent_covid_initial_states(n_simulants, initial_states.loc[draw])

        df_counts = pd.DataFrame(index=beta.loc[start_time:].index, columns=['S', 'E', 'I1', 'I2', 'R', 'new_infections'])
        for t in df_counts.index[:-1]:
            df_counts.loc[t] = df.covid_state.value_counts()
            df_counts.loc[t, 'new_infections'] = agent_covid_step(df, beta=beta.loc[t, draw], **params[draw])

        df_counts.loc[df_counts.index[-1]] = df.covid_state.value_counts()

        df_count_list.append(df_counts)

    return df_count_list


