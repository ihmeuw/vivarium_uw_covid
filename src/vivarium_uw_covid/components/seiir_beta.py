"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd


def beta_predict(coeffs, covs):
    """Predict beta(t) values from coefficients
    and covariates
    
    Parameters
    ----------
    coeffs : pd.DataFrame with columns for covariates and rows for draws
             e.g. the output of load_effect_coefficients
    covs : pd.DataFrame with columns for covariates and rows for each
           time for a specific location, e.g the output of load_covariates, e.g.
           results of load_covariates filtered to a specific location::
               df_covs = cs.load_covariates(cov_dir)
               covs = df_covs[df_covs.location_id == loc_id]

    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each day for location specified loc_id
    """

    log_beta = np.zeros((len(covs), 1_000))
    for i in coeffs.columns:
        if i == 'intercept':
            log_beta += coeffs[i].values  # .values is not necessary for some vesion of pandas
        else:
            log_beta += np.outer(covs[i], coeffs[i])
    return pd.DataFrame(np.exp(log_beta), index=covs.index)


def make_alternative_covariates(df_covs, loc_id, **alt_cov_values):
    """Create alternative scenario beta(t) values for specified location
        
    Parameters
    ----------
    df_covs : pd.DataFrame with columns for covariates and rows for each
              time, e.g the output of load_covariates
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    alt_cov_values : covariate names and new values, which can be floats, array-like, or pd.Series,
                     e.g. testing_reference=0.005, mask_use=1.0 to create a scenario with high mask
                     use and high testing
    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each day for location specified loc_id
    """
    alt_covs = df_covs[df_covs.location_id == loc_id].copy()
    for cov, val in alt_cov_values.items():
        alt_covs[cov] = val

    return alt_covs
