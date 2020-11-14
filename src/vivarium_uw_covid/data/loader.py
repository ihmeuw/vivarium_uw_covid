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

from vivarium import Artifact
from vivarium_uw_covid import paths, globals as project_globals
import json


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


UW_DATA_DIR = '/ihme/homes/abie/notebook/2020/data'

def load_covariates(cov_dir, location_id):
    """Load all covariate values
    
    Parameters
    ----------
    cov_dir : str, a directory in /ihme/covid-19/seir-covariates/,
              e.g. 'best'
    
    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each location for each time
    """
    cov_list = ['testing', 'mask_use', 'mobility', 'proportion_over_1k',  'proportion_over_5k',  # HACK: start with something that is a dataframe to get the index set
                'air_pollution_pm_2_5', 'lri_mortality', 'pneumonia',
                'proportion_over_2_5k', 'proportion_under_100m', 'smoking_prevalence',
                'temperature']

    df_covs = pd.DataFrame()
    for cov in cov_list:
        t = pd.read_csv(f'/ihme/covid-19/seir-covariates/{cov_dir}/{cov}/reference_scenario.csv')
        t = t.set_index('location_id')
        tt = t.loc[location_id]
        if isinstance(tt, pd.DataFrame):
            tt = tt.set_index('date')
        df_covs[cov] = tt[f'{cov}_reference']

    df_covs.index = df_covs.index.map(pd.Timestamp)
    return df_covs


def load_effect_coefficients(run_dir, loc_id):
    """Load 1,000 draws of effect coefficients (from beta regression)
    
    Parameters
    ----------
    run_dir : str, a directory in /ihme/covid-19/seir-regression/,
              e.g. 'best'
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    
    Results
    -------
    returns pd.DataFrame with columns for covariates,
    rows for each draw
    """
    coeffs = {}
    for draw in range(1_000):
        df_coeffs = pd.read_csv(f'/ihme/covid-19/seir-regression/{run_dir}/coefficients/draw_{draw}.csv')
        s_coeffs = df_coeffs.set_index('location_id').loc[loc_id]
        coeffs[draw] = s_coeffs
    coeffs = pd.DataFrame(coeffs).T
    return coeffs


def load_beta_fit(run_dir, loc_id):
    """Load 1,000 draws of beta from SEIIR fit (the step before beta regression)
    
    Parameters
    ----------
    run_dir : str, a directory in /ihme/covid-19/seir-regression/,
              e.g. 'best'
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    
    Results
    -------
    returns pd.DataFrame formed by appending dataframes for all draws, with column for indicating which draw
    """
    betas = []
    for draw in range(1_000):
        df_fit = pd.read_csv(f'/ihme/covid-19/seir-regression/{run_dir}/beta/draw_{draw}.csv')
        df_fit = df_fit.set_index('location_id').loc[loc_id]

        df_fit['draw'] = draw
        betas.append(df_fit)
    return pd.concat(betas)


def load_seiir_compartment_sizes(run_dir, loc_id):
    """Load 1,000 draws of compartment sizes
    
    Parameters
    ----------
    run_dir : str, a directory in /ihme/covid-19/seir-regression/,
              e.g. 'best'
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    
    Results
    -------
    returns pd.DataFrame with columns for each state,
    rows for each draw
    """

    compartment_sizes = []
    for draw in range(1_000):
        df_proj = pd.read_csv(f'/ihme/covid-19/seir-forecast/{run_dir}/reference/component_draws/draw_{draw}.csv')
        df_proj = df_proj.set_index('location_id').loc[loc_id]

        df_proj.index = df_proj.pop('date').map(pd.Timestamp)
        df_proj['draw'] = draw
        compartment_sizes.append(df_proj)

    return pd.concat(compartment_sizes)


def load_seiir_params(run_dir, theta):
    """Load 1,000 draws of seiir model parameters
    
    Parameters
    ----------
    run_dir : str, a directory in /ihme/covid-19/seir-regression/,
              e.g. 'best'
    theta : float > 0, the value of the theta parameter, which is not stored
            the param_draw csv
    
    Results
    -------
    returns dict-of-dicts where each draw is the key for a 
    dicts of parameters values
    """
    params = {}
    for draw in range(1_000):
        draw = str(draw)
        df_params = pd.read_csv(f'/ihme/covid-19/seir-regression/{run_dir}/parameters/draw_{draw}.csv')
        params[draw] = df_params.set_index('params')['values'].to_dict()
        params[draw].pop('day_shift')  # TODO: find out what this is, and if I should be using it
        params[draw]['theta'] = theta
    
    params = pd.DataFrame(params)
    return params


def extract_covid_projection_data(art_fname, run_dir, loc_id):
    """Extract and transform all data necessary for covid projections in a
    single location, and store in a Vivarium Artifact
    
    Parameters
    ----------
    art_fname : str, path for vivarium artifact to hold transformed data
    run_dir : str, a directory in /ihme/covid-19/seir-regression/,
              e.g. 'best'
    loc_id : int, a location id, e.g. 60886 for "King and Snohomish Counties", described in e.g.
             /ihme/covid-19/model-inputs/best/locations/covariate_with_aggregates_hierarchy.csv
    
    Results
    -------
    returns Vivarium Artifact
    """
    cov_dir = 'best'
    rates_dir = 'best'
    loc_id = int(loc_id)

    art = Artifact(art_fname)

    metadata_dict = dict(cov_dir=cov_dir, run_dir=run_dir, rates_dir=rates_dir, loc_id=loc_id)
    art.write('metadata.data_params', metadata_dict)


    df_covs = load_covariates(cov_dir, loc_id)
    assert len(df_covs) > 0
    art.write('beta.covariates', df_covs)

    coeffs = load_effect_coefficients(run_dir, loc_id)
    art.write('beta.coeffs', coeffs)
    beta_fit = load_beta_fit(run_dir, loc_id)
    art.write('beta.fit', beta_fit)

    compartment_sizes = load_seiir_compartment_sizes(run_dir, loc_id)
    art.write('seiir.compartment_sizes', compartment_sizes)
    # TODO: get a composite initial state by mixing initial states from multiple locations

    # extract theta from initial states
    theta = compartment_sizes.theta.mean()
    assert compartment_sizes.theta.std() == 0, 'so far theta has been a fixed value; investigate if that changes'
    params = load_seiir_params(run_dir, theta)
    art.write('seiir.params', params)

    df_fac_staff = load_uw_fac_staff_ages()
    df_ifr = load_ifr(rates_dir)
    art.write('covid_deaths.fac_staff_ages', df_fac_staff)
    art.write('covid_deaths.ifr', df_ifr)

    return art


def load_uw_fac_staff_ages():
    """Load age/sex distribution for UW faculty and staff
    
    Results
    -------
    returns pd.DataFrame with columns for age_start, age_end, sex, and proportion ('p')
    """

    df_fac_staff = pd.read_csv(f'{UW_DATA_DIR}/uw_staff_age_sex_counts.csv')
    df_fac_staff['p'] = df_fac_staff.value / df_fac_staff.value.sum()
    return df_fac_staff.filter(['age_start', 'age_end', 'sex', 'p'])


def load_ifr(rates_dir):
    """Load age-specific infection fatality ratio

    Parameters
    ----------
    rates_dir : str, a directory in /ihme/covid-19/rates,
              e.g. 'best'
    
    Results
    -------
    return pd.DataFrame
    """
    df_ifr = pd.read_csv(f'/ihme/covid-19/rates/{rates_dir}/ifr_preds_1yr.csv')
    return df_ifr


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
