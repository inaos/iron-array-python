from skbuild import setup

DESCRIPTION = 'The Math Array Accelerator for Python'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="iarray",
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    python_requires=">=3.6",
    #package_dir={'': '.'},
    package_dir={'iarray': 'iarray'},
    packages=['iarray', 'iarray.py2llvm', 'iarray.tests'],
    package_data={
        'iarray': ['libsvml.so', 'libintlc.so', 'libintlc.so.5', 'libiarray.so'],
    },
    extras_require={
        'doc': [
            'sphinx >= 1.5',
            'sphinx_rtd_theme',
            'numpydoc',
        ],
        'examples': [
            'matplotlib',
            'numexpr',
            'numba',
        ]},
)
