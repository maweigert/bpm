import os
from setuptools import setup

setup(name='bpm',
    version='0.1.1',
    description='renders spim data in 3D',
    url='http://mweigert@bitbucket.org/mweigert/bpm',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=['bpm'],
    install_requires=[
        'numpy', 'scipy',
    ],

    package_data={"bpm":['kernels/*','psf_integrals/*.cl',]},

)
