"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd
from typing import Tuple, Union

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
    returns pd.DataFrame of beta values, with columns for each draw,
    and rows for each day
    """

    log_beta = np.zeros((len(covs), 1_000))
    for i in coeffs.columns:
        if i == 'intercept':
            log_beta += coeffs[i].values  # .values is not necessary for some vesion of pandas
        else:
            log_beta += np.outer(covs[i], coeffs[i])
    return pd.DataFrame(np.exp(log_beta), index=covs.index)


# from https://github.com/ihmeuw/covid-model-seiir-pipeline/blob/master/src/covid_model_seiir_pipeline/core/utils.py#L128
def beta_shift(beta_fit: pd.DataFrame,
               beta_pred: np.ndarray,
               window_size: Union[int, None] = None,
               avg_over: Union[int, None] = None) -> Tuple[np.ndarray, float]:
    """Calculate the beta shift.
    Args:
        beta_fit (pd.DataFrame): Data frame contains the date and beta fit.
        beta_pred (np.ndarray): beta prediction.
        windown_size (Union[int, None], optional):
            Window size for the transition. If `None`, Hard shift no transition.
            Default to None.
        avg_over (Union[int, None], optional):
            Final beta scale depends on the ratio between beta prediction over the
            average beta over `avg_over` days. If `None`, final scale will be 1, means
            return to the `beta_pred` completely. Default to None.
    Returns:
        Tuple[np.ndarray, float]: Predicted beta, after scaling (shift) and the initial scaling.
    """
    assert 'date' in beta_fit.columns, "'date' has to be in beta_fit data frame."
    assert 'beta' in beta_fit.columns, "'beta' has to be in beta_fit data frame."
    beta_fit = beta_fit.sort_values('date')
    beta_fit = beta_fit['beta'].values

    anchor_beta = beta_fit[-1]
    scale_init = anchor_beta / beta_pred[0]

    if avg_over is None:
        scale_final = 1.0
    else:
        beta_history = beta_fit[-avg_over:]
        scale_final = beta_history.mean() / beta_pred[0]

    if window_size is not None:
        assert isinstance(window_size, int) and window_size > 0, f"window_size={window_size} has to be a positive " \
                                                                 f"integer."
        scale = scale_init + (scale_final - scale_init)/window_size*np.arange(beta_pred.size)
        scale[(window_size + 1):] = scale_final
    else:
        scale = scale_init

    betas = beta_pred * scale

    return betas, scale_init


def beta_finalize(beta_pred, beta_fit):
    """Calculate the final beta(t) values from beta_pred and beta_fit
    
    Parameters
    ----------
    beta_pred : pd.DataFrame with columns for covariates and rows for draws
                e.g. the output of beta_predict
    beta_fit : dict of pd.DataFrames with columns for beta and date,
               e.g the output of load_beta_fit

    Results
    -------
    returns pd.DataFrame of final beta values, with columns for each draw,
    and rows for each day
    """
    beta_final = pd.DataFrame(columns=beta_pred.columns, index=beta_pred.index)
    crossover_time = pd.Timestamp(beta_fit[0].date.max())
    for draw in beta_fit.keys():
        s_final, scale_init = beta_shift(beta_fit=beta_fit[draw],
                                         beta_pred=beta_pred.loc[crossover_time:, draw],
                                         window_size=28,
                                         avg_over=21)
        beta_final.loc[:, draw] = s_final
    return beta_final


def make_alternative_covariates(df_covs, loc_id, start_time, **alt_cov_values):
    """Create alternative scenario beta(t) values for specified location
        
    Parameters
    ----------
    df_covs : pd.DataFrame with columns for covariates and rows for each
              time, e.g the output of load_covariates
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    start_time : pd.Timestamp
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
        alt_covs.loc[start_time:, cov] = val

    return alt_covs
