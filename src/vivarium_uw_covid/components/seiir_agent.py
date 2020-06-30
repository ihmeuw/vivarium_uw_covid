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
    n_infectious = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants = len(df)
    infection_rate = (beta * n_infectious**alpha + theta) / n_simulants

    return agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta)

def agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta,
                                         use_mechanistic_testing=False, test_rate=.001, test_positive_rate=.05):
    """Make one step for agent-based model

    Parameters
    ----------
    TODO: additional docstring parameters
    use_mechanistic_testing : bool
    test_rate : tests per person per day
    test_positive_rate : fraction of daily tests that test positive (if there are enough infections to do so)

    Results
    -------
    returns number of agents infected during this time step
    """
    uniform_random_draw = np.random.uniform(size=len(df))

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

    pr_S_to_E = 1 - np.exp(-infection_rate)
    rows = (df.covid_state == 'S') & (uniform_random_draw < pr_S_to_E)
    df.loc[rows, 'covid_state'] = 'E'
    new_infections = np.sum(rows)

    #### code for mechanistic testing-and-isolation model
    #### TODO: refactor into a separate Vivarium component
    if use_mechanistic_testing:
        n_test_positive = test_rate * test_positive_rate * len(df)
        n_infected = ((df.covid_state == 'E') | (df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()

        if n_infected > 0:
            test_rate_among_infected = n_test_positive / n_infected
            if test_rate_among_infected > 10:
                pr_tested = 1
            else:
                pr_tested = 1 - np.exp(-test_rate_among_infected)
    
            rows = (df.covid_state != 'S') & (np.random.uniform(size=len(df)) < pr_tested)  # FIXME: detected too soon?
            df.loc[rows, 'covid_state'] = 'R'  # move any non-S state individual to state R if they are tested (FIXME: too simple)


    return new_infections


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


