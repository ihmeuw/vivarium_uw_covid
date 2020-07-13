import numpy as np, pandas as pd
import vivarium_uw_covid as vuc

def test_sample_covid_deaths():
    df = vuc.initialize_age_and_sex(n_fac_staff=1_000, n_student=1_000)
    death = vuc.sample_covid_deaths(df)

    assert sum(death) > 0

