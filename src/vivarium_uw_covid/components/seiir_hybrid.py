"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd
from .seiir_compartmental import compartmental_covid_step
from .seiir_agent import agent_covid_initial_states, agent_covid_step_with_infection_rate


def compartmental_hybrid_step(s0, s_ip, mixing_parameter, **params):
    """Find the compartment sizes for the compartmental part of a hybrid model

    Parameters
    ----------
    s0 : counts for compartmental part
    s_ip : counts for individual part
    addl params : transmission parameters (alpha, beta, gamma1, gamma2, sigma, theta)
    """
    n_simulants = (s0.S + s0.E + s0.I1 + s0.I2 + s0.R) + mixing_parameter * (s_ip.S + s_ip.E + s_ip.I1 + s_ip.I2 + s_ip.R)
    n_infectious = (s0.I1 + s0.I2) + mixing_parameter * (s_ip.I1 + s_ip.I2)
    
    s1 = compartmental_covid_step(s0, n_simulants, n_infectious, **params)
    
    return s1

               
def individual_hybrid_step(df, s0, mixing_parameter, alpha, beta, gamma1, gamma2, sigma, theta):
    """ df : population table for individuals
    s0 : compartment sizes for outside population
    addl parameters : transmission parameters
    """
    n_infectious = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants = len(df)
    
    n_infectious_out = (s0.I1 + s0.I2)
    n_simulants_out = (s0.S + s0.E + s0.I1 + s0.I2 + s0.R)
    infection_rate = ((1 - mixing_parameter) * (beta * n_infectious**alpha + theta) / n_simulants + mixing_parameter * beta * n_infectious_out**alpha / n_simulants_out)

    return agent_covid_step_with_infection_rate(df, infection_rate, alpha, gamma1, gamma2, sigma, theta)
               

def run_hybrid_model(n_draws, n_simulants, mixing_parameter, params, beta_agent, beta_compartment, start_time, initial_states):
    """Project population sizes from start time to end of beta.index
    
    Parameters
    ----------
    n_draws : int
    n_simulants : int
    mixing_parameter : float >= 0 and <= 1, signifying the fraction of time the
                       agents spend with the outside population (vs with other agents)
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
    df_agent_count_list, df_compartment_count_list = [], []

    for draw in np.random.choice(range(1_000), replace=False, size=n_draws):
        ## initialize population table for individual-based model
        df_individual = pd.DataFrame(index=range(n_simulants))
        df_individual['covid_state'] = agent_covid_initial_states(n_simulants, initial_states.loc[draw])


        ## initialize counts table for inidividual and compartmental models
        df_individual_counts = pd.DataFrame(index=beta_agent.loc[start_time:].index, columns=['S', 'E', 'I1', 'I2', 'R', 'new_infections'])

        df_compartment = pd.DataFrame(index=beta_compartment.loc[start_time:].index, columns=['S', 'E', 'I1', 'I2', 'R', 'new_infections'])
        #### initialize states from IHME Projection for time zero
        for state in ['S', 'E', 'I1', 'I2', 'R']:
            df_compartment.loc[start_time, state] = initial_states.loc[draw, state]


        ## step through hybrid simulation
        dt = pd.Timedelta(days=1)
        for t in df_individual_counts.index[:-1]:
            df_individual_counts.loc[t] = df_individual.covid_state.value_counts()

            df_compartment.loc[t+dt] = compartmental_hybrid_step(
                                            df_compartment.loc[t], df_individual_counts.loc[t],
                                            beta=beta_compartment.loc[t, draw],
                                            mixing_parameter=mixing_parameter, **params[draw])

            df_individual_counts.loc[t, 'new_infections'] = individual_hybrid_step(df_individual,
                                        df_compartment.loc[t], beta=beta_agent.loc[t, draw],
                                        mixing_parameter=mixing_parameter, **params[draw])

        df_individual_counts.loc[df_individual_counts.index[-1]] = df_individual.covid_state.value_counts()

        df_agent_count_list.append(df_individual_counts)
        df_compartment_count_list.append(df_compartment)
               
    return df_agent_count_list, df_compartment_count_list

