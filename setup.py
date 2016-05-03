import os
from setuptools import setup, find_packages

setup(name='beamy',
    version='0.1.1',
    description='beam propagation',
    url='',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=['bpm'],
    install_requires=[
        'numpy', 'scipy',"pyopencl>=2015.2.4"
    ],

    package_data={"bpm":['psf/kernels/*.cl','bpm/kernels/*.cl',]},

)
