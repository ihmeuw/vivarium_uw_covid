import numpy as np, pandas as pd
import vivarium_uw_covid as vuc

from vivarium import Artifact
art = Artifact('src/vivarium_uw_covid/artifacts/wisc.hdf')
df_fac_staff = art.load('covid_deaths.fac_staff_ages')
df_ifr = art.load('covid_deaths.ifr')

def test_sample_covid_deaths():
    np.random.seed(12345)
    df = vuc.initialize_age_and_sex(df_fac_staff, n_fac_staff=1_000, n_student=1_000)
    f_ifr = vuc.initialize_ifr(df_ifr)
    death = vuc.sample_covid_deaths(df, f_ifr)

    assert sum(death) > 0


def test_generate_covid_deaths():
    df = pd.DataFrame({'R': [100_000, 200_000]})

    deaths00 = vuc.generate_covid_deaths(df_fac_staff, df_ifr, {0:df}, start_date=0, end_date=1, student_frac=0.0)
    deaths05 = vuc.generate_covid_deaths(df_fac_staff, df_ifr, {0:df}, start_date=0, end_date=1, student_frac=0.5)
    deaths10 = vuc.generate_covid_deaths(df_fac_staff, df_ifr, {0:df}, start_date=0, end_date=1, student_frac=1.0)

    assert float(deaths00) >= float(deaths05)
    assert float(deaths05) >= float(deaths10)

