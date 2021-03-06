import itertools

from typing import NamedTuple

####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_uw_covid'
CLUSTER_PROJECT = 'proj_cost_effect'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '3G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    # TODO - project locations here
]


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    STRUCTURE: str = 'population.structure'
    AGE_BINS: str = 'population.age_bins'
    DEMOGRAPHY: str = 'population.demographic_dimensions'
    TMRLE: str = 'population.theoretical_minimum_risk_life_expectancy'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'

    @property
    def name(self):
        return 'population'

    @property
    def log_name(self):
        return 'population'


POPULATION = __Population()


# TODO - sample key group used to idneitfy keys in model
# For more information see the tutorial:
# https://vivarium-inputs.readthedocs.io/en/latest/tutorials/pulling_data.html#entity-measure-data
class __IHD(NamedTuple):
    ACUTE_MI_PREVALENCE: str = 'sequela.acute_myocardial_infarction.prevalence'
    POST_MI_PREVALENCE: str = 'sequela.post_myocardial_infarction.prevalence'
    ACUTE_MI_INCIDENCE_RATE: str = 'cause.ischemic_heart_disease.incidence_rate'
    ACUTE_MI_DISABILITY_WEIGHT: str = 'sequela.acute_myocardial_infarction.disability_weight'
    POST_MI_DISABILITY_WEIGHT: str = 'sequela.post_myocardial_infarction.disability_weight'
    ACUTE_MI_EMR: str = 'sequela.acute_myocardial_infarction.excess_mortality_rate'
    POST_MI_EMR: str = 'sequela.post_myocardial_infarction.excess_mortality_rate'
    CSMR: str = 'cause.ischemic_heart_disease.cause_specific_mortality_rate'
    RESTRICTIONS: str = 'cause.ischemic_heart_disease.restrictions'

    @property
    def name(self):
        return 'ischemic_heart_disease'

    @property
    def log_name(self):
        return 'ischemic heart disease'


IHD = __IHD()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    IHD,
]


###########################
# Disease Model variables #
###########################


class TransitionString(str):

    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split('_TO_')
        return obj


# TODO input details of model states and transitions
SOME_MODEL_NAME = 'some_model'
SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{SOME_MODEL_NAME}'
FIRST_STATE_NAME = 'first_state'
SECOND_STATE_NAME = 'second_state'
IHD_MODEL_STATES = (SUSCEPTIBLE_STATE_NAME, FIRST_STATE_NAME, SECOND_STATE_NAME)
IHD_MODEL_TRANSITIONS = (
    TransitionString(f'{SUSCEPTIBLE_STATE_NAME}_TO_{FIRST_STATE_NAME}'),
    TransitionString(f'{FIRST_STATE_NAME}_TO_{SECOND_STATE_NAME}'),
    TransitionString(f'{SECOND_STATE_NAME}_TO_{FIRST_STATE_NAME}')
)


########################
# Risk Model Constants #
########################
# TODO - remove if you don't need lbwsg
LBWSG_MODEL_NAME = 'low_birth_weight_and_short_gestation'


class __LBWSG_MISSING_CATEGORY(NamedTuple):
    CAT: str = 'cat212'
    NAME: str = 'Birth prevalence - [37, 38) wks, [1000, 1500) g'
    EXPOSURE: float = 0.


LBWSG_MISSING_CATEGORY = __LBWSG_MISSING_CATEGORY()


#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'
OUTPUT_SCENARIO_COLUMN = 'scenario'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
PERSON_TIME_COLUMN_TEMPLATE = 'person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
TRANSITION_COUNT_COLUMN_TEMPLATE = '{TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'

COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
    'person_time': PERSON_TIME_COLUMN_TEMPLATE,
    'deaths': DEATH_COLUMN_TEMPLATE,
    'ylls': YLLS_COLUMN_TEMPLATE,
    'ylds': YLDS_COLUMN_TEMPLATE,
    'state_person_time': STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'transition_count': TRANSITION_COUNT_COLUMN_TEMPLATE,
}

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
# TODO - add literals for years in the model
YEARS = ()
# TODO - add literals for ages in the model
AGE_GROUPS = ()
# TODO - add causes of death
CAUSES_OF_DEATH = (
    'other_causes',
#    DIARRHEA_WITH_CONDITION_STATE_NAME,
)
# TODO - add causes of disability
CAUSES_OF_DISABILITY = tuple()
#    DIARRHEA_WITH_CONDITION_STATE_NAME,
#)
STATES = tuple() #(state for model in DISEASE_MODELS for state in DISEASE_MODEL_MAP[model]['states'])
TRANSITIONS = tuple() #(transition for model in DISEASE_MODELS for transition in DISEASE_MODEL_MAP[model]['transitions'])

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_GROUP': AGE_GROUPS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
#    'STATE': STATES,
#    'TRANSITION': TRANSITIONS,
}


def RESULT_COLUMNS(kind='all'):
    if kind not in COLUMN_TEMPLATES and kind != 'all':
        raise ValueError(f'Unknown result column type {kind}')
    columns = []
    if kind == 'all':
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {field: values
                              for field, values in TEMPLATE_FIELD_MAP.items() if f'{{{field}}}' in template}
        fields, value_groups = filtered_field_map.keys(), itertools.product(*filtered_field_map.values())
        for value_group in value_groups:
            columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}))
    return columns

