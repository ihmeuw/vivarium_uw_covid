#!/usr/bin/env python
import os

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_uw_covid", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'vivarium==0.9.3',
        'vivarium_public_health==0.10.4',

        'click',
        'gbd_mapping==2.1.0',
        'jinja2',
        'loguru',
        'numpy<=1.15.4',
        'pandas<0.25',
        'scipy',
        'tables<=3.4.0',
        'pyyaml',
    ]

    # use "pip install -e .[dev]" to install required components + extra components
    extras_require = [
        'vivarium_cluster_tools==1.1.2',
        'vivarium_inputs[data]==3.1.1',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        extras_require={
            'dev': extras_require,
        },

        zip_safe=False,

        entry_points='''
            [console_scripts]
            make_specs=vivarium_uw_covid.tools.cli:make_specs
            make_artifacts=vivarium_uw_covid.tools.cli:make_artifacts
        '''
    )
