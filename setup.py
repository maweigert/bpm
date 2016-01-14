import os
from setuptools import setup, find_packages

setup(name='bpm',
    version='0.1.1',
    description='beam propagation',
    url='',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=['bpm'],
    install_requires=[
        'numpy', 'scipy',
    ],

    package_data={"bpm":['psf/kernels/*','psf_integrals/*.cl',]},

)
