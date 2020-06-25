"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd


def compartmental_covid_step(s0, n_simulants, n_infectious, alpha, beta, gamma1, gamma2, sigma, theta):
    """Project next-day population sizes from current-day populations and model parameters
    
    Parameters
    ----------
    s0 : pd.Series with counts for S, E, I1, I2, and R
    n_simulants : float
    n_infections : float
    alpha, beta, gamma1, gamma2, sigma, theta : float
    
    Results
    -------
    returns pd.Series with counts for S, E, I1, I2, and R
    
    Notes
    -----
    This is intended to replicate the SEIIR ODE in the IHME Projection Model, which is listed
    here [1], but has been augmented to include the importation parameter theta::
    
        ds = -beta*(s/self.N)*(i1 + i2)**self.alpha - theta/N
        de = beta*(s/self.N)*(i1 + i2)**self.alpha - self.sigma*e
        di1 = self.sigma*e - self.gamma1*i1 + theta/N  # TODO: get IHME Projection model to change theta/N addition to e instead of i1
        di2 = self.gamma1*i1 - self.gamma2*i2
        dr = self.gamma2*i2

    [1] https://github.com/ihmeuw-msca/ODEOPT/blob/master/src/odeopt/ode/system/nonlinearsys.py#L215-L219
    """

    s1 = s0.copy()

    assert theta >= 0, 'only handle theta >= 0 for now'
    pr_infected = 1 - np.exp(-(beta * n_infectious**alpha + theta) / n_simulants) ### * .85)  # FIXME: why does *.85 help??
    dS = np.random.binomial(s0.S, pr_infected)
    s1.new_infections = dS
    s1.S -= dS
    s1.E += dS

    pr_E_to_I1 = 1 - np.exp(-sigma)
    dE = np.random.binomial(s0.E, pr_E_to_I1)
    s1.E -= dE
    s1.I1 += dE

    pr_I1_to_I2 = 1 - np.exp(-gamma1)
    dI1 = np.random.binomial(s0.I1, pr_I1_to_I2)
    s1.I1 -= dI1
    s1.I2 += dI1

    pr_I2_to_R = 1 - np.exp(-gamma2)
    dI2 = np.random.binomial(s0.I2, pr_I2_to_R)
    s1.I2 -= dI2
    s1.R += dI2

    return s1


def run_compartmental_model(n_draws, n_simulants, params, beta, start_time, initial_states):
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
    days = beta.loc[start_time:].index
    compartments = ['S', 'E', 'I1', 'I2', 'R', 'new_infections']
    
    df_count_list = []
    for draw in np.random.choice(range(1_000), replace=False, size=n_draws):
        df_i = pd.DataFrame(index=days, columns=compartments)

        # initialize states from IHME Projection for time zero
        for state in compartments:
            if state != 'new_infections':
                df_i.loc[start_time, state] = initial_states.loc[draw, state]
        df_i.loc[start_time] *= n_simulants / df_i.loc[start_time].sum()  # rescale to have requested number of simulants
    
        dt = pd.Timedelta(days=1)
        for t in df_i.index[:-1]:
            s0 = df_i.loc[t]
            n_simulants = (s0.S + s0.E + s0.I1 + s0.I2 + s0.R)
            n_infectious = (s0.I1 + s0.I2)
            df_i.loc[t+dt] = compartmental_covid_step(s0, n_simulants=n_simulants, n_infectious=n_infectious,
                                                      beta=beta.loc[t, draw], **params[draw])
        df_count_list.append(df_i)
    return df_count_list

