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

    Results
    -------
    returns np.array with .shape == (n_simulants,) and values S, E, I1, I2, and R
    chosen randomly with expected fractions matching compartment_sizes
    """
    state_list = ['S', 'E', 'I1', 'I2', 'R']
    
    p = [compartment_sizes[state] for state in state_list]
    p = np.array(p)
    p /= p.sum()
    
    state = np.random.choice(state_list, size=n_simulants, replace=True, p=p)
    return state


def agent_covid_step(df, alpha, beta, gamma1, gamma2, sigma, theta):
    """Make one step for agent-based model

    Parameters
    ----------
    df : pd.DataFrame of population table for agents: each row is an individual,
         and column 'covid_state' indicates which individuals are S, E, I1, I2, and R
    alpha, beta, gamma1, gamma2, sigma, theta : parameter values for infectious disease dynamics

    Results
    -------
    updates the population table based on one day of infectious disease dynamics,
    returns number of agents in each state after one step
    """
    n_infectious = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants = len(df)
    infection_rate = (beta * n_infectious**alpha + theta) / n_simulants

    return agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta)

def agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta,
                                         use_mechanistic_testing=False, test_rate=.001, test_positive_rate=.05):
    """Make one step for agent-based model, after infection rate has been calculated

    Parameters
    ----------
    df : pd.DataFrame of population table for agents: each row is an individual,
         and column 'covid_state' indicates which individuals are S, E, I1, I2, and R
    infection_rate : float, 0 <= infection_rate < 1
    alpha, beta, gamma1, gamma2, sigma, theta : parameter values for infectious disease dynamics
    use_mechanistic_testing : bool
    test_rate : tests per person per day
    test_positive_rate : fraction of daily tests that test positive (if there are enough infections to do so)

    Results
    -------
    updates the population table based on one day of infectious disease dynamics,
    returns number of agents in each state after one step
    """
    substeps=3
    dt = 1/substeps
    n_new_infections = 0
    for i in range(substeps):
        uniform_random_draw = np.random.uniform(size=len(df))

        # update from R back to S, to allow in-place computation
        pr_I2_to_R = 1 - np.exp(-dt*gamma2)
        rows = (df.covid_state == 'I2') & (uniform_random_draw < pr_I2_to_R)
        df.loc[rows, 'covid_state'] = 'R'

        pr_I1_to_I2 = 1 - np.exp(-dt*gamma1)
        rows = (df.covid_state == 'I1') & (uniform_random_draw < pr_I1_to_I2)
        df.loc[rows, 'covid_state'] = 'I2'

        pr_E_to_I1 = 1 - np.exp(-dt*sigma)
        rows = (df.covid_state == 'E') & (uniform_random_draw < pr_E_to_I1)
        df.loc[rows, 'covid_state'] = 'I1'

        pr_S_to_E = 1 - np.exp(-dt*infection_rate)
        rows = (df.covid_state == 'S') & (uniform_random_draw < pr_S_to_E)
        df.loc[rows, 'covid_state'] = 'E'
        n_new_infections += np.sum(rows)

    #### code for mechanistic testing-and-isolation model
    #### TODO: refactor into a separate Vivarium component
    n_new_isolations = 0
    if use_mechanistic_testing:
        n_test_positive = test_rate * test_positive_rate * len(df)
        n_infected = ((df.covid_state == 'E') | (df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()

        if n_infected > 0:
            test_rate_among_infected = n_test_positive / n_infected
            if test_rate_among_infected > 10:
                pr_tested = 1
            else:
                pr_tested = 1 - np.exp(-test_rate_among_infected)
    
            rows = (df.covid_state.isin(['I1', 'I2'])) & (np.random.uniform(size=len(df)) < pr_tested)  # FIXME: detected too soon?
            df.loc[rows, 'covid_state'] = 'R'  # move any non-S state individual to state R if they are tested (FIXME: too simple)
            n_new_isolations = np.sum(rows)

    s_result = pd.Series(df.covid_state.value_counts(), index=['S', 'E', 'I1', 'I2', 'R', 'n_new_infections', 'n_new_isolations'])
    s_result = s_result.fillna(0)
    s_result['n_new_infections'] = n_new_infections
    s_result['n_new_isolations'] = n_new_isolations
    return s_result


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

        df_counts = pd.DataFrame(index=beta.loc[start_time:].index, columns=['S', 'E', 'I1', 'I2', 'R', 'n_new_infections', 'n_new_isolations'])
        df_counts.iloc[0] = df.covid_state.value_counts()
        for t in df_counts.index[1:]:
            df_counts.loc[t] = agent_covid_step(df, beta=beta.loc[t, draw], **params[draw])

        df_count_list.append(df_counts)

    return df_count_list


