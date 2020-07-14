from pathlib import Path
import vivarium_uw_covid
import vivarium_uw_covid.globals as project_globals

BASE_DIR = Path(vivarium_uw_covid.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{project_globals.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / 'model_specifications'
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{project_globals.PROJECT_NAME}/')

SEIIR_DIR = '/ihme/covid-19/seir-pipeline-outputs'
RATES_DIR = '/ihme/covid-19/rates'
UW_DATA_DIR = '/home/j/Project/simulation_science/covid/data'
