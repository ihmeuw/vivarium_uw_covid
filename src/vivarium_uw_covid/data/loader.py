"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
from vivarium_gbd_access import gbd
from gbd_mapping import causes, risk_factors, covariates
import pandas as pd
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import interface, utilities, utility_data, globals as vi_globals
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_uw_covid import paths, globals as project_globals


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        project_globals.POPULATION.STRUCTURE: load_population_structure,
        project_globals.POPULATION.AGE_BINS: load_age_bins,
        project_globals.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        project_globals.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        project_globals.POPULATION.ACMR: load_standard_data,

        # TODO - add appropriate mappings
        # project_globals.DIARRHEA_PREVALENCE: load_standard_data,
        # project_globals.DIARRHEA_INCIDENCE_RATE: load_standard_data,
        # project_globals.DIARRHEA_REMISSION_RATE: load_standard_data,
        # project_globals.DIARRHEA_CAUSE_SPECIFIC_MORTALITY_RATE: load_standard_data,
        # project_globals.DIARRHEA_EXCESS_MORTALITY_RATE: load_standard_data,
        # project_globals.DIARRHEA_DISABILITY_WEIGHT: load_standard_data,
        # project_globals.DIARRHEA_RESTRICTIONS: load_metadata,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    return interface.get_population_structure(location)


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location)


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    return interface.get_measure(entity, key.measure, location)


def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = get_entity(key)
    metadata = entity[key.measure]
    if hasattr(metadata, 'to_dict'):
        metadata = metadata.to_dict()
    return metadata


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = utilities.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = utilities.reshape(data)
    data = utilities.scrub_gbd_conventions(data, location)
    data = utilities.split_interval(data, interval_column='age', split_column_prefix='age')
    data = utilities.split_interval(data, interval_column='year', split_column_prefix='year')
    return utilities.sort_hierarchical_data(data)


from ..paths import SEIIR_DIR

def load_covariates(cov_dir):
    """Load all covariate values
    
    Parameters
    ----------
    cov_dir : str, a directory in $SEIIR_DIR/covariate/,
              e.g. '2020_06_23.03.01'
    
    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each location for each time
    """
    
    df_covs = pd.read_csv(f'{SEIIR_DIR}/covariate/{cov_dir}/cached_covariates.csv', index_col=0)
    df_covs.index = df_covs.date.map(pd.Timestamp)
    return df_covs


def load_effect_coefficients(run_dir, loc_id):
    """Load 1,000 draws of effect coefficients (from beta regression)
    
    Parameters
    ----------
    run_dir : str, a directory in $SEIIR_DIR/regression/,
              e.g. '2020_06_23.07'
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    
    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each draw
    """
    coeffs = {}
    for draw in range(1_000):
        df_coeffs = pd.read_csv(f'{SEIIR_DIR}/regression/{run_dir}/coefficients/coefficients_{draw}.csv', index_col=0)
        s_coeffs = df_coeffs.set_index('group_id').loc[loc_id].sort_values()
        coeffs[draw] = s_coeffs
    coeffs = pd.DataFrame(coeffs).T
    return coeffs


def load_seiir_initial_states(run_dir, loc_id, start_date):
    """Load 1,000 draws of compartment sizes on specified date,
    to use as initial states for simulation
    
    Parameters
    ----------
    run_dir : str, a directory in $SEIIR_DIR/regression/,
              e.g. '2020_06_23.07'
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    start_date : pd.Timestamp, e.g. pd.Timestamp('2020-09-01')
    
    Results
    -------
    returns pd.DataFrame with columns for each state,
    rows for each draw
    """

    initial_states = {}
    for draw in range(1_000):
        df_proj = pd.read_csv(f'{SEIIR_DIR}/forecast/{run_dir}/component_draws/{loc_id}/draw_{draw}.csv', index_col=0)
        df_proj.index = df_proj.pop('date').map(pd.Timestamp)
        s_proj = df_proj.loc[start_date]
        initial_states[draw] = s_proj
    initial_states = pd.DataFrame(initial_states).T
    return initial_states


def load_seiir_params(run_dir, theta):
    """Load 1,000 draws of seiir model parameters
    
    Parameters
    ----------
    run_dir : str, a directory in $SEIIR_DIR/regression/,
              e.g. '2020_06_23.07'
    theta : float > 0, the value of the theta parameter, which is not stored
            the param_draw csv
    
    Results
    -------
    returns dict-of-dicts where each draw is the key for a 
    dicts of parameters values
    """
    params = {}
    for draw in range(1_000):
        df_params = pd.read_csv(f'{SEIIR_DIR}/regression/{run_dir}/parameters/params_draw_{draw}.csv')
        params[draw] = df_params.set_index('params')['values'].to_dict()
        params[draw].pop('day_shift')  # TODO: find out what this is, and if I should be using it
        params[draw]['theta'] = theta
    
    return params



def get_entity(key: str):
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]
