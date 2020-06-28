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

               
def individual_hybrid_step(df, s_compartment, mixing_parameter, alpha, beta_agent, beta_compartment, gamma1, gamma2, sigma, theta):
    """ df : population table for individuals
    s_compartment : compartment sizes for outside population
    addl parameters : transmission parameters
    """
    n_infectious_agent = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants_agent = len(df)
    
    n_infectious_compartment = (s_compartment.I1 + s_compartment.I2)
    n_simulants_compartment = (s_compartment.S + s_compartment.E + s_compartment.I1 + s_compartment.I2 + s_compartment.R)
    infection_rate = ((1 - mixing_parameter) * (beta_agent * n_infectious_agent**alpha + theta) / n_simulants_agent
                      + mixing_parameter * beta_compartment * n_infectious_compartment**alpha / n_simulants_compartment)

    return agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta)
               

def run_hybrid_model(n_draws, n_simulants, mixing_parameter, params,
                     beta_agent, beta_compartment,
                     start_time,
                     initial_states_agent, initial_states_compartment):
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
    
    Results
    -------
    returns list of pd.DataFrames with colunms for counts for S, E, I1, I2, and R
    and rows for each day of projection
    """
    df_agent_count_list, df_compartment_count_list = [], []

    days = beta_agent.loc[start_time:].index
    states = ['S', 'E', 'I1', 'I2', 'R']

    for draw in np.random.choice(range(1_000), replace=False, size=n_draws):
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
        for t in df_individual_counts.index[:-1]:
            df_individual_counts.loc[t] = df_individual.covid_state.value_counts()

            df_compartment.loc[t+dt] = compartmental_hybrid_step(
                                            df_compartment.loc[t], df_individual_counts.loc[t],
                                            beta=beta_compartment.loc[t, draw],
                                            mixing_parameter=mixing_parameter, **params[draw])

            df_individual_counts.loc[t, 'new_infections'] = individual_hybrid_step(df_individual,
                                        df_compartment.loc[t],
                                        beta_agent=beta_agent.loc[t, draw],
                                        beta_compartment=beta_compartment.loc[t, draw],
                                        mixing_parameter=mixing_parameter, **params[draw])

        # store last day of counts from the agent states
        df_individual_counts.loc[df_individual_counts.index[-1]] = df_individual.covid_state.value_counts()

        # append the counts to their lists
        df_agent_count_list.append(df_individual_counts)
        df_compartment_count_list.append(df_compartment)
               
    return df_agent_count_list, df_compartment_count_list

