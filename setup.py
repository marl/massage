""" Setup script for massage. """
from setuptools import setup

import imp

version = imp.load_source('massage.version', 'massage/version.py')

if __name__ == "__main__":
    setup(
        name='massage',

        version=version.version,

        description='Multitrack Analysis/SynthesiS for Annotation, auGmentation and Evaluation',

        author='Rachel Bittner, Justin Salamon',

        author_email='rachel.bittner@nyu.edu, justin.salamon@nyu.edu',

        url='https://github.com/justinsalamon/massage',

        download_url='https://github.com/justinsalamon/massage/releases',

        packages=['massage'],

        package_data={'massage': []},

        long_description="""Multitrack Analysis/SynthesiS for Annotation, auGmentation and Evaluation""",

        keywords='augmentation music synthesis multitrack pitch',

        license='MIT',

        dependency_links=['git+ssh://git@github.com/marl/medleydb.git'],

        install_requires=[
            'six',
            'medleydb == 1.3.2patch',
            'numpy >= 1.8.0',
            'scipy >= 0.13.0',
            'librosa >= 0.5.0',
            'vamp >= 1.1.0',
            'sox >= 1.3.0',
            'jams >= 0.3.0',
            'pyfluidsynth >= 1.2.4',
            'pretty_midi >= 0.2.7'
        ],

        extras_require={
            'tests': [
                'mock',
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
