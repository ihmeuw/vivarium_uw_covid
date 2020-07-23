import numpy as np, pandas as pd
from vivarium import Artifact
from vivarium_uw_covid.components.seiir_beta import *


art = Artifact('src/vivarium_uw_covid/artifacts/new_york.hdf')
df_covs = art.load('beta.covariates')
coeffs = art.load('beta.coeffs')
params = art.load('seiir.params')
compartment_sizes = art.load('seiir.compartment_sizes')
t0 = '2020-09-01'
initial_states = compartment_sizes.loc[t0].set_index('draw').dropna()


def test_beta_predict():
    beta_pred = beta_predict(coeffs, df_covs)


def test_make_alternative_covariates():
    df_alt = make_alternative_covariates(df_covs, t0, mask_use=.99)
    assert not np.allclose(df_alt.mask_use, .99)
    assert np.allclose(df_alt.mask_use.loc[t0:], .99)


def test_beta_finalize():
    beta_pred = beta_predict(coeffs, df_covs)
    beta_fit  = art.load('beta.fit')
    beta_final = beta_finalize(beta_pred, beta_fit[beta_fit.draw == 0])  # just test for one draw of beta_fit, because that is faster

    assert pd.Timestamp(beta_fit[beta_fit.draw == 0].date.iloc[-1]) == beta_final[0].dropna().index[0], 'beta_final non-nan values should start on the last day of beta_fit'
    assert beta_pred.index[-1] == beta_final.index[-1], 'beta_final should end on the last day of beta_pred'


