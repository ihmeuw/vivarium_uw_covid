"""Methods for modeling COVID mortality,
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-07-08
"""

import numpy as np, pandas as pd
import scipy.interpolate, pymc as pm

def f_ifr_factory(df_ifr, logit_shift):
    """Create age-interpolating IFR function

    Parameters
    ----------
    df_ifr : pd.DataFrame with columns for age_mid, lowest_ifr
    logit_shift : float, shift of value in logit-space

    Results
    -------
    returns function that maps from age to IFR
    """
    return scipy.interpolate.interp1d(df_ifr.age_mid.values,
                                      pm.invlogit(df_ifr.lowest_ifr+logit_shift),
                                      kind='linear', fill_value='extrapolate')

def initialize_ifr(df_ifr):
    """Create sex-specific IFR age-interpolation functions
    
    Parameters
    ----------
    df_ifr : pd.DataFrame with columns for age_mid and lowest_ifr

    Results
    -------
    returns dict with keys male and female and values that are functions from age to IFR
    """

    # logit_shift value from Reed,
    # https://ihme.slack.com/archives/C0138B6810W/p1594319840211400?thread_ts=1594236461.208800&cid=C0138B6810W
    f_ifr = {'male': f_ifr_factory(df_ifr, +0.305),
             'female': f_ifr_factory(df_ifr, -0.305),
         }

    return f_ifr


def initialize_age_and_sex(df_fac_staff, n_fac_staff, n_student):
    """Create age and sex columns for a population table

    Parameters
    ----------
    df_fac_staff : pd.DataFrame with columns age_start, age_end, sex, and 'p' (proportion in this strata)
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


def calculate_ifr(df, f_ifr):
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


def sample_covid_deaths(df, f_ifr):
    """Determine which individuals die from COVID
    for each person in the population table

    Parameters
    ----------
    df : pd.DataFrame with a row for each person, and colunms for age and sex

    Results
    -------
    returns a pd.Series of boolean values (True means the individual died due to COVID)
    """

    ifr = calculate_ifr(df, f_ifr)
    return (np.random.uniform(size=len(df)) <= ifr)


def generate_one_draw_of_covid_deaths(df, df_fac_staff, start_date, end_date, student_frac, f_ifr, seed):
    """Generate estimated cumulative count of individuals to die from COVID
    for a single draw of simulation output on selected date

    Parameters
    ----------
    df : pd.DataFrames with a row for each day of sim, and 
              a column "R" for the removed individuals
    df_fac_staff : pd.DataFrame
    start_date, end_date : pd.Timestamp in index of dataframes in df_dict, to be used for start and end date of cumulative count
    student_frac : float in interval (0,1), to be used for mix of student/non-students
    f_irf : dict of interpolated age-specific ifr functions for males and females
    seed : int, random seed for reproducible random numbers

    Results
    -------
    returns count of deaths
    """
    np.random.seed(13245+seed)
    n_infected = int(np.round(df.loc[end_date, 'R'] - df.loc[start_date, 'R']))
    n_student = np.random.binomial(n_infected, student_frac)
    n_fac_staff = n_infected - n_student
    df_ages = initialize_age_and_sex(df_fac_staff, n_fac_staff, n_student)
    s_death = sample_covid_deaths(df_ages, f_ifr)
    return s_death.sum()


def generate_covid_deaths(df_fac_staff, df_ifr, df_dict, start_date, end_date, student_frac):
    """Generate estimated cumulative count of individuals to die from COVID
    for each simulation output on selected date

    Parameters
    ----------
    df_dict : dict of pd.DataFrames with a row for each day of sim, and 
              a column "R" for the removed individuals
    start_date, end_date : pd.Timestamp in index of dataframes in df_dict, to be used for start and end date of cumulative count
    student_frac : float in interval (0,1), to be used for mix of student/non-students

    Results
    -------
    returns a pd.Series of death counts
    """
    from dask import delayed, compute
    
    f_ifr = initialize_ifr(df_ifr)

    results_dict = {}
    for draw, df in df_dict.items():
        results_dict[draw] = delayed(generate_one_draw_of_covid_deaths)(df, df_fac_staff,
                                                start_date, end_date, student_frac, f_ifr, draw)

    results_tuple = compute(results_dict)  # dask.compute returns a 1-tuple (FIXME: unless n_draws == 1)
    return pd.Series(results_tuple[0]) # entry 0 of the 1-tuple is the dict
