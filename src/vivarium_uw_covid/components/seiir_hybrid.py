"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd
from .seiir_compartmental import compartmental_covid_step
from .seiir_agent import agent_covid_initial_states, agent_covid_step_with_infection_rate


def compartmental_hybrid_step(s_compartment, s_agent, mixing_parameter, **params):
    """Find the compartment sizes for the compartmental part of a hybrid model

    Parameters
    ----------
    s_compartment : counts for compartmental part
    s_agent : counts for individual part
    addl params : transmission parameters (alpha, beta, gamma1, gamma2, sigma, theta)

    Results
    -------
    return pd.Series of counts for compartmental part of hybrid model on next day
    """
    s_agent = s_agent.fillna(0)  # make sure counts are zero and not np.nan

    n_simulants = ((s_compartment.S + s_compartment.E + s_compartment.I1 + s_compartment.I2 + s_compartment.R)
                   + mixing_parameter * (s_agent.S + s_agent.E + s_agent.I1 + s_agent.I2 + s_agent.R))
    n_infectious = (s_compartment.I1 + s_compartment.I2) + mixing_parameter * (s_agent.I1 + s_agent.I2)
    
    s1 = compartmental_covid_step(s_compartment, n_simulants, n_infectious, **params)
    
    return s1

               
def individual_hybrid_step(df, s_compartment, mixing_parameter, alpha,
                           beta_agent, beta_compartment,
                           gamma1, gamma2, sigma, theta,
                           use_mechanistic_testing=False, test_rate=.001, test_positive_rate=.05):
    """Make one step for hybrid model

    Parameters
    ----------
    TODO: additional docstring parameters
    df : population table for individuals
    s_compartment : compartment sizes for outside population
    addl parameters : transmission parameters
    use_mechanistic_testing : bool
    test_rate : tests per person per day
    test_positive_rate : fraction of daily tests that test positive (if there are enough infections to do so)
    """
    n_infectious_agent = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants_agent = len(df)
    
    n_infectious_compartment = (s_compartment.I1 + s_compartment.I2)
    n_simulants_compartment = (s_compartment.S + s_compartment.E + s_compartment.I1 + s_compartment.I2 + s_compartment.R)
    infection_rate = ((1 - mixing_parameter) * (beta_agent * n_infectious_agent**alpha + theta) / n_simulants_agent
                      + mixing_parameter * beta_compartment * n_infectious_compartment**alpha / n_simulants_compartment)

    return agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta,
                                                use_mechanistic_testing, test_rate, test_positive_rate)
               

def run_one_hybrid_model(draw, n_simulants, mixing_parameter, params,
                         beta_agent, beta_compartment,
                         start_time,
                         initial_states_agent, initial_states_compartment,
                         use_mechanistic_testing=False, test_rate=.001, test_positive_rate=.05):
    """Project population sizes from start time to end of beta.index
    
    Parameters
    ----------
    draws : int
    n_simulants : int
    mixing_parameter : float >= 0 and <= 1, signifying the fraction of time the
                       agents spend with the outside population (vs with other agents)
    params : dict-of-dicts, where draw maps to dict with values for 
                alpha, gamma1, gamma2, sigma, theta : float
    beta_agent : pd.DataFrame with index of dates and columns for draws
    beta_compartment : pd.DataFrame with index of dates and columns for draws
    start_time : pd.Timestamp
    initial_states_agent : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R
    initial_states_compartment : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R
    use_mechanistic_testing : bool
    test_rate : tests per person per day
    test_positive_rate : fraction of daily tests that test positive (if there are enough infections to do so)

    Results
    -------
    returns list of pd.DataFrames with colunms for counts for S, E, I1, I2, and R
    as well as new infections, and rows for each day of projection
    """

    states = ['S', 'E', 'I1', 'I2', 'R']
    days = beta_agent.loc[start_time:].index

    ## initialize population table for individual-based model
    df_individual = pd.DataFrame(index=range(n_simulants))
    df_individual['covid_state'] = agent_covid_initial_states(n_simulants, initial_states_agent.loc[draw])

    ## initialize counts table for inidividual and compartmental models
    df_individual_counts = pd.DataFrame(index=days, columns=states + ['new_infections'])

    df_compartment = pd.DataFrame(index=days, columns=states + ['new_infections'])
    #### initialize compartmental model state sizes for time zero
    for state in states:
        df_compartment.loc[start_time, state] = initial_states_compartment.loc[draw, state]


    ## step through hybrid simulation
    dt = pd.Timedelta(days=1)
    df_individual_counts.iloc[0] = df_individual.covid_state.value_counts()
    for t in df_individual_counts.index[:-1]:

        df_compartment.loc[t+dt] = compartmental_hybrid_step(
                                        df_compartment.loc[t], df_individual_counts.loc[t],
                                        beta=beta_compartment.loc[t, draw],
                                        mixing_parameter=mixing_parameter, **params[draw])

        df_individual_counts.loc[t+dt] = individual_hybrid_step(df_individual,
                                    df_compartment.loc[t],
                                    beta_agent=beta_agent.loc[t, draw],
                                    beta_compartment=beta_compartment.loc[t, draw],
                                    mixing_parameter=mixing_parameter,
                                    use_mechanistic_testing=use_mechanistic_testing,
                                    test_rate=test_rate, test_positive_rate=test_positive_rate,
                                    **params[draw])

    return df_individual_counts, df_compartment



def run_hybrid_model(n_draws, n_simulants, mixing_parameter, params,
                     beta_agent, beta_compartment,
                     start_time,
                     initial_states_agent, initial_states_compartment,
                     use_mechanistic_testing=False, test_rate=.001, test_positive_rate=.05):
    """Project population sizes from start time to end of beta.index
    
    Parameters
    ----------
    n_draws : int
    n_simulants : int
    mixing_parameter : float >= 0 and <= 1, signifying the fraction of time the
                       agents spend with the outside population (vs with other agents)
    params : dict-of-dicts, where draw maps to dict with values for 
                alpha, gamma1, gamma2, sigma, theta : float
    beta_agent : pd.DataFrame with index of dates and columns for draws
    beta_compartment : pd.DataFrame with index of dates and columns for draws
    start_time : pd.Timestamp
    initial_states_agent : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R
    initial_states_compartment : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R
    use_mechanistic_testing : bool
    test_rate : tests per person per day
    test_positive_rate : fraction of daily tests that test positive (if there are enough infections to do so)

    Results
    -------
    returns list of pd.DataFrames with colunms for counts for S, E, I1, I2, and R
    as well as new infections, and rows for each day of projection
    """
    assert 0 <= mixing_parameter <= 1, 'mixing_parameter must be in interval [0,1]'

    df_agent_count_list, df_compartment_count_list = [], []

    for draw in np.random.choice(range(1_000), replace=False, size=n_draws):
        
        df_individual_counts, df_compartment = run_one_hybrid_model(draw, n_simulants, mixing_parameter, params,
                                                                    beta_agent, beta_compartment,
                                                                    start_time,
                                                                    initial_states_agent, initial_states_compartment,
                                                                    use_mechanistic_testing, test_rate, test_positive_rate)


        # append the counts to their lists
        df_agent_count_list.append(df_individual_counts)
        df_compartment_count_list.append(df_compartment)
               
    return df_agent_count_list, df_compartment_count_list


def prun_hybrid_model(n_draws, n_simulants, mixing_parameter, params,
                     beta_agent, beta_compartment,
                     start_time,
                     initial_states_agent, initial_states_compartment,
                     use_mechanistic_testing=False, test_rate=.001, test_positive_rate=.05):
    """Project population sizes from start time to end of beta.index
    
    Parameters
    ----------
    n_draws : int
    n_simulants : int
    mixing_parameter : float >= 0 and <= 1, signifying the fraction of time the
                       agents spend with the outside population (vs with other agents)
    params : dict-of-dicts, where draw maps to dict with values for 
                alpha, gamma1, gamma2, sigma, theta : float
    beta_agent : pd.DataFrame with index of dates and columns for draws
    beta_compartment : pd.DataFrame with index of dates and columns for draws
    start_time : pd.Timestamp
    initial_states_agent : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R
    initial_states_compartment : pd.DataFrame with index of draws and colunms for S, E, I1, I2, R
    use_mechanistic_testing : bool
    test_rate : tests per person per day
    test_positive_rate : fraction of daily tests that test positive (if there are enough infections to do so)

    Results
    -------
    returns list of pd.DataFrames with colunms for counts for S, E, I1, I2, and R
    as well as new infections, and rows for each day of projection
    """
    from dask import delayed, compute

    assert 0 <= mixing_parameter <= 1, 'mixing_parameter must be in interval [0,1]'

    df_agent_count_list, df_compartment_count_list = [], []

    for draw in np.random.choice(range(1_000), replace=False, size=n_draws):
        
        df_tuple = delayed(run_one_hybrid_model)(draw, n_simulants, mixing_parameter, params,
                                                 beta_agent, beta_compartment,
                                                 start_time,
                                                 initial_states_agent, initial_states_compartment,
                                                 use_mechanistic_testing, test_rate, test_positive_rate)


        # append the counts to their lists
        df_agent_count_list.append(df_tuple[0])
        df_compartment_count_list.append(df_tuple[1])
               
    return compute(df_agent_count_list, df_compartment_count_list)



