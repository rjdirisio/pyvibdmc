PyVibDMC
==============================
[//]: # (Badges)
[![codecov](https://codecov.io/gh/rjdirisio/PyVibDMC/branch/master/graph/badge.svg)](https://codecov.io/gh/rjdirisio/PyVibDMC/branch/master)
![pytest](https://github.com/rjdirisio/pyvibdmc/workflows/CI_pytest/badge.svg)
[![DOI](https://zenodo.org/badge/281503719.svg)](https://zenodo.org/badge/latestdoi/281503719)


A general purpose diffusion monte carlo code for studying vibrational problems

This package requires the following:

- NumPy

- Matplotlib

- h5py

- A potential energy surface (PES) for a system of interest, which can be called using a Python function 
(See Documentation).

- Optional: MPI4Py (for multi-node PES evaluation, otherwise uses multiprocessing for multi-core PES evaluation)
  
- Optional: Tensorflow (for Neural Network PES)

- Tutorial: A compiler required for the potential energy surface (the tutorial potential uses gfortran)

- Tutorial: make (on Linux systems, this is usually installed via the 'build-essential' or 'Development Tools' packages )

### Documentation

Visit the Documentation hosted on [ReadTheDocs](https://pyvibdmc.readthedocs.io/en/latest/)

### Installation

You may view the latest stable release on the [Python Package Index](https://pypi.org/project/pyvibdmc/).

You may install it through `pip`:

`pip install pyvibdmc`


### Contributing

Features should be developed on branches. To create and switch to a branch, use the command

`git checkout -b new_branch_name`

To switch to an existing branch, use

`git checkout branch_name`


### Copyright

Copyright (c) 2020, Ryan DiRisio


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.

Thank you to the entire McCoy group for helping me talk through this code, with special acknowledgements to Fenris Lu (beta tester), Mark Boyer (coding conversations), and my advisor, Anne McCoy.
