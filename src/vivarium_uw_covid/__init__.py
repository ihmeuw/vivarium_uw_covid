"""vivarium_uw_covid

Research repository for the vivarium_uw_covid project.

"""

from .components.seiir_beta import *

from .components.seiir_compartmental import *
from .components.seiir_agent import *
from .components.seiir_hybrid import *
from .components.covid_deaths import *

from .tools.plots import *


def data_dict_from_artifact(fname):
    """Create data dictionary from Vivarium artifact

    Parameters
    ----------
    fname : str, path to vivarium artifact (an hdf file)

    Results
    -------
    return python dict of data to use in simulation

    Notes
    -----
    This is a hacky function for convenience, and perhaps should be
    removed in the future
    """
    print(f'Loading all COVID-19 projection data from {fname}')

    from vivarium import Artifact
    art = Artifact(fname)
    data =  art.load('metadata.data_params')
    data.update({
        'coeffs': art.load('beta.coeffs'),
        'df_covs': art.load('beta.covariates'),
        'beta_fit': art.load('beta.fit'),

        'compartment_sizes': art.load('seiir.compartment_sizes'),
        'params': art.load('seiir.params'),
        'df_fac_staff': art.load('covid_deaths.fac_staff_ages'),
        'df_ifr': art.load('covid_deaths.ifr'),
    })

    return data
