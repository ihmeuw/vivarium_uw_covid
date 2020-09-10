"""Methods for instantiating and running a hybrid microsimulation
calibrated to IHME COVID Projections

Author: Abraham D. Flaxman
Date: 2020-06-25
"""

import numpy as np, pandas as pd
from typing import Tuple, Union, Dict

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


# from https://github.com/ihmeuw/covid-model-seiir-pipeline/blob/develop/src/covid_model_seiir_pipeline/core/utils.py#L137-L210
def beta_shift(beta_fit: pd.DataFrame,
               beta_pred: np.ndarray,
               draw_id: int,
               window_size: Union[int, None] = None,
               average_over_min: int = 1,
               average_over_max: int = 35,
               total_deaths: int = None,
               offset: float = 0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calculate the beta shift.
    Args:
        beta_fit (pd.DataFrame): Data frame contains the date and beta fit.
        beta_pred (np.ndarray): beta prediction.
        draw_id (int): Draw of data provided.  Will be used as a seed for
            a random number generator to determine the amount of beta history
            to leverage in rescaling the y-intercept for the beta prediction.
        window_size (Union[int, None], optional):
            Window size for the transition. If `None`, Hard shift no transition.
            Default to None.
    Returns:
        Tuple[np.ndarray, float]: Predicted beta, after scaling (shift) and the initial scaling.
    """
    assert 'date' in beta_fit.columns, "'date' has to be in beta_fit data frame."
    assert 'beta' in beta_fit.columns, "'beta' has to be in beta_fit data frame."
    beta_fit = beta_fit.sort_values('date')
    beta_hat = beta_fit['beta_pred'].values
    beta_fit = beta_fit['beta'].values
    one_week = 7  # minimum period over which to residual average
    deaths_upper = 300  # bound at which we let the residual average be what it is
    deaths_lower = 150  # bound at which we shift the residual average so the mean across draws is 0

    rs = np.random.RandomState(seed=draw_id)
    a = rs.randint(1, average_over_min)
    b = rs.randint(a + one_week, average_over_max)

    beta_fit_final = beta_fit[-1]
    beta_pred_start = beta_pred[0]

    if total_deaths is None or deaths_upper <= total_deaths:
        offset = 0
    elif deaths_lower <= total_deaths < deaths_upper:
        offset_scale = (deaths_upper - total_deaths)/(deaths_upper - deaths_lower)
        offset *= offset_scale
    else:  # total_deaths < deaths_lower, use the provided offset (the mean across draws)
        pass

    scale_init = beta_fit_final / beta_pred_start
    log_beta_resid = np.log(beta_fit / beta_hat)
    residual_mean = log_beta_resid[-b:-a].mean()
    scale_final = np.exp(residual_mean - offset)

    scale_params = {
        'window_size': window_size,
        'history_days_end': b,
        'history_days_start': a,
        'fit_final': beta_fit_final,
        'pred_start': beta_pred_start,
        'beta_ratio_mean': scale_final,
        'offset': offset,
        'beta_residual_mean': np.log(scale_final),
    }

    if window_size is not None:
        assert isinstance(window_size, int) and window_size > 0, f"window_size={window_size} has to be a positive " \
                                                                 f"integer."
        scale = scale_init + (scale_final - scale_init)/window_size*np.arange(beta_pred.size)
        scale[(window_size + 1):] = scale_final
    else:
        scale = scale_init

    betas = beta_pred * scale

    return betas, scale_params


def beta_finalize(beta_pred, beta_fit):
    """Calculate the final beta(t) values from beta_pred and beta_fit
    
    Parameters
    ----------
    beta_pred : pd.DataFrame with columns for covariates and rows for draws
                e.g. the output of beta_predict
    beta_fit : pd.DataFrame with columns for beta and date and draw,
               e.g the output of load_beta_fit

    Results
    -------
    returns pd.DataFrame of final beta values, with columns for each draw,
    and rows for each day
    """
    assert len(beta_pred.index.unique()) == len(beta_pred.index), 'expect no duplicate values in index of beta_pred'
    beta_final = pd.DataFrame(columns=beta_pred.columns, index=beta_pred.index)
    crossover_time = pd.Timestamp(beta_fit.date.max())
    for draw in beta_fit.draw.unique():
        s_final, scale_params = beta_shift(
            beta_fit=beta_fit[beta_fit.draw==draw],
            beta_pred=beta_pred.loc[crossover_time:, draw].values, # Peng likes numpy arrays
            draw_id=draw,
            window_size=42,
            average_over_min=7,
            average_over_max=42
        )
        beta_final.loc[crossover_time:, draw] = s_final
    return beta_final


def make_alternative_covariates(df_covs, start_time, **alt_cov_values):
    """Create alternative scenario beta(t) values for specified location
        
    Parameters
    ----------
    df_covs : pd.DataFrame with columns for covariates and rows for each
              time, e.g the output of load_covariates
    start_time : pd.Timestamp
    alt_cov_values : covariate names and new values, which can be floats, array-like, or pd.Series,
                     e.g. testing_reference=0.005, mask_use=1.0 to create a scenario with high mask
                     use and high testing; if name starts with "delta_" then add to value, instead
                     of replacing it; if name starts with "pct_delta_" then multiply by value/100,
                     instead of replacing it; if name starts with "max_" then use this to clip value,
                     instead of replacing it.
    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each day
    """
    alt_covs = df_covs.copy()
    for cov, val in alt_cov_values.items():
        if cov.startswith('delta_'):
            cov = cov.replace('delta_', '')
            alt_covs.loc[start_time:, cov] += val
        if cov.startswith('pct_delta_'):
            cov = cov.replace('pct_delta_', '')
            alt_covs.loc[start_time:, cov] *= val/100
        elif cov.startswith('max_'):
            cov = cov.replace('max_', '')
            alt_covs.loc[start_time:, cov] = np.minimum(val, alt_covs.loc[start_time:, cov])
        else:
            alt_covs.loc[start_time:, cov] = val

    return alt_covs


def make_beta(coeffs, df_covs, beta_fit):
    """Create alternative scenario beta(t) values for specified location
        
    Parameters
    ----------
    coeffs : pd.DataFrame with columns for covariates and rows for draws
             e.g. the output of load_effect_coefficients
    df_covs : pd.DataFrame with columns for covariates and rows for each
              time, e.g the output of load_covariates
    beta_fit : pd.DataFrame with columns for beta and date and draw,
               e.g the output of load_beta_fit
    Results
    -------
    returns pd.DataFrame of final beta values, with columns for each draw,
    and rows for each day
    """
    beta_pred = beta_predict(coeffs, df_covs)
    beta_final = beta_finalize(beta_pred, beta_fit)
    return beta_final
