import numpy as np, pandas as pd
import vivarium_uw_covid as vuc

df_fac_staff = vuc.load_uw_fac_staff_ages()
df_ifr = vuc.load_ifr('2020_06_29.01')

def test_sample_covid_deaths():
    df = vuc.initialize_age_and_sex(df_fac_staff, n_fac_staff=1_000, n_student=1_000)
    f_ifr = vuc.initialize_ifr(df_ifr)
    death = vuc.sample_covid_deaths(df, f_ifr)

    assert sum(death) > 0


def test_generate_covid_deaths():
    df = pd.DataFrame({'R': [100_000]})
    f_ifr = vuc.initialize_ifr(df_ifr)

    deaths00 = vuc.generate_covid_deaths(df_fac_staff, f_ifr, [df], end_date=0, student_frac=0.0)
    deaths05 = vuc.generate_covid_deaths(df_fac_staff, f_ifr, [df], end_date=0, student_frac=0.5)
    deaths10 = vuc.generate_covid_deaths(df_fac_staff, f_ifr, [df], end_date=0, student_frac=1.0)

    assert float(deaths00) >= float(deaths05)
    assert float(deaths05) >= float(deaths10)

