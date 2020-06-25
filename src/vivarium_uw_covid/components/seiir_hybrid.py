"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd
#from .seiir_compartment import 
from .seiir_agent import agent_covid_initial_states


def compartmental_hybrid_step(s0, s_ip, alpha, beta, gamma1, gamma2, sigma, theta):
    """Find the compartment sizes for the compartmental part of a hybrid model

    Parameters
    ----------
    s0 : counts for compartmental part
    s_ip : counts for individual part
    addl params : transmission parameters
    """
    s1 = s0.copy()
    
    n_simulants = (s0.S + s0.E + s0.I1 + s0.I2 + s0.R) + (s_ip.S + s_ip.E + s_ip.I1 + s_ip.I2 + s_ip.R)
    n_infectious = (s0.I1 + s0.I2) + (s_ip.I1 + s_ip.I2)
    
    ### TODO: refactor this so it does not repeat the code from the non-hybrid compartmental model
    pr_infected = 1 - np.exp(-(beta * n_infectious**alpha + theta) / n_simulants)
    dS = s0.S*pr_infected
    s1.S -= dS
    s1.E += dS
    s1.new_infections = dS
    
    pr_E_to_I1 = 1 - np.exp(-sigma)
    dE = s0.E*pr_E_to_I1
    s1.E -= dE
    s1.I1 += dE

    
    pr_I1_to_I2 = 1 - np.exp(-gamma1)
    dI1 = s0.I1*pr_I1_to_I2
    s1.I1 -= dI1
    s1.I2 += dI1
    
    
    pr_I2_to_R = 1 - np.exp(-gamma2)
    dI2 = s0.I2*pr_I2_to_R
    s1.I2 -= dI2
    s1.R += dI2

    
    return s1

               
def individual_hybrid_step(df, s0, alpha, beta, gamma1, gamma2, sigma, theta):
    """ df : population table for individuals
    s0 : compartment sizes for outside population
    addl parameters : transmission parameters
    """
    
    # find the number infectious
    n_infectious = ((df.covid_state == 'I1') | (df.covid_state == 'I2')).sum()
    n_simulants = len(df)
    
    n_infectious_out = (s0.I1 + s0.I2)
    n_simulants_out = (s0.S + s0.E + s0.I1 + s0.I2 + s0.R)
    
    # update from R back to S, to allow in-place computation
    uniform_random_draw = np.random.uniform(size=n_simulants)
    pr_I2_to_R = 1 - np.exp(-gamma2)
    rows = (df.covid_state == 'I2') & (uniform_random_draw < pr_I2_to_R)
    df.loc[rows, 'covid_state'] = 'R'

    pr_I1_to_I2 = 1 - np.exp(-gamma1)
    rows = (df.covid_state == 'I1') & (uniform_random_draw < pr_I1_to_I2)
    df.loc[rows, 'covid_state'] = 'I2'
    
    pr_E_to_I1 = 1 - np.exp(-sigma)
    rows = (df.covid_state == 'E') & (uniform_random_draw < pr_E_to_I1)
    df.loc[rows, 'covid_state'] = 'I1'
    
    mixing_parameter = .5 # addl parameter for how much time students are mixed with rest of population
    pr_infected = 1 - np.exp(-((1 - mixing_parameter) * beta * (n_infectious**alpha + theta) / n_simulants
                              + mixing_parameter * beta * n_infectious_out**alpha / n_simulants_out))
    rows = (df.covid_state == 'S') & (uniform_random_draw < pr_infected)
    df.loc[rows, 'covid_state'] = 'E'

    return np.sum(rows)
               

def run_hybrid_model(n_draws, n_simulants, params, beta_agent, beta_compartment, start_time, initial_states):
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
                                            beta=beta_compartment.loc[t, draw], **params[draw])

            df_individual_counts.loc[t, 'new_infections'] = individual_hybrid_step(df_individual, df_compartment.loc[t], beta=beta_agent.loc[t, draw], **params[draw])

        df_individual_counts.loc[df_individual_counts.index[-1]] = df_individual.covid_state.value_counts()

        df_agent_count_list.append(df_individual_counts)
        df_compartment_count_list.append(df_compartment)
               
    return df_agent_count_list, df_compartment_count_list

