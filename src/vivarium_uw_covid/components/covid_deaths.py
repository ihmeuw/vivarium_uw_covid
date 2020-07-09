"""Methods for modeling COVID mortality,
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-07-08
"""

import numpy as np, pandas as pd
import scipy.interpolate, pymc as pm


# TODO: refactor this into loader.py
dir_name = '/home/j/Project/simulation_science/covid/data'
df_fac_staff = pd.read_csv(f'{dir_name}/uw_staff_age_sex_counts.csv')
df_fac_staff['p'] = df_fac_staff.value / df_fac_staff.value.sum()

rate_dir = '2020_06_29.01'
df = pd.read_csv(f'/ihme/covid-19/rates/{rate_dir}/ifr_preds_1yr.csv')
f_ifr_ = scipy.interpolate.interp1d(df.age_mid.values, pm.invlogit(df.lowest_ifr), kind='linear', fill_value='extrapolate')
f_ifr = {'male': f_ifr_,  # TODO: include sex ratio
         'female': f_ifr_,
     }


def initialize_age_and_sex(n_fac_staff, n_student):
    """Create age and sex columns for a population table

    Parameters
    ----------
    n_fac_staff : int, number of faculty/staff to include in the population
    n_students : int, number of students to include in the population

    Results
    -------
    returns pd.DataFrame with n_fac_staff + n_student rows, and columns
    for age and sex
    """
    weighted_random_rows = np.random.choice(df_fac_staff.index, size=n_fac_staff, p=df_fac_staff.p)
    
    df1 = pd.DataFrame(index=range(n_fac_staff))
    age_start = df_fac_staff.age_start.loc[weighted_random_rows].values
    age_end = df_fac_staff.age_end.loc[weighted_random_rows].values
    df1['age'] = np.random.uniform(low=age_start, high=age_end)
    df1['sex'] = df_fac_staff.sex.loc[weighted_random_rows].values
    
    df2 = pd.DataFrame(index=range(n_student))
    df2['age'] = np.random.uniform(low=18, high=22, size=n_student)
    df2['sex'] = np.random.choice(['male', 'female'], size=n_student)
    return pd.concat([df1, df2]).reset_index(drop=True)


def calculate_ifr(df):
    """Calculate the infection fatality ratio (IFR)
    for each person in the population table

    Parameters
    ----------
    df : pd.DataFrame with a row for each person, and colunms for age and sex

    Results
    -------
    returns a pd.Series of IFR values
    """
    ifr = pd.Series(index=df.index)
    
    for sex, df_sex in df.groupby('sex'):
        f_ifr_sex = f_ifr[sex]
        ifr.loc[df_sex.index] = f_ifr_sex(df_sex.age)
    return ifr


def sample_covid_deaths(df):
    """Determine which individuals die from COVID
    for each person in the population table

    Parameters
    ----------
    df : pd.DataFrame with a row for each person, and colunms for age and sex

    Results
    -------
    returns a pd.Series of boolean values (True means the individual died due to COVID)
    """

    ifr = calculate_ifr(df)
    return (np.random.uniform(size=len(df)) <= ifr)


def generate_covid_deaths(df_list, end_date):
    """Generate estimated cumulative count of individuals to die from COVID
    for each simulation output on selected date

    Parameters
    ----------
    df_list : list of pd.DataFrames with a row for each day of sim, and 
    a column "R" for the removed individuals

    Results
    -------
    returns a pd.Series of death counts
    """
    deaths = []
    for df in df_list:
        n_infected = df.loc[end_date, 'R']
        n_fac_staff, n_student = int(np.floor(n_infected/2)), int(np.ceil(n_infected/2))
        df_ages = initialize_age_and_sex(n_fac_staff, n_student)
        s_death = sample_covid_deaths(df_ages)
        
        deaths.append(s_death.sum())
    return pd.Series(deaths)
