PyVibDMC
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/rjdirisio/PyVibDMC.svg?branch=master)](https://travis-ci.com/rjdirisio/PyVibDMC)
[![codecov](https://codecov.io/gh/rjdirisio/PyVibDMC/branch/master/graph/badge.svg)](https://codecov.io/gh/rjdirisio/PyVibDMC/branch/master)


A general purpose diffusion monte carlo code for studying vibrational problems

This repository is currently under development. To do a development, install, download this repository and then type

`pip install -e .`

in the repository directory.

This package requires the following:

-numpy
-matplotlib
-A potential energy surface (PES) for a system of interest, which can be called using a Python function (Documentation pending on this).

Features should be developed on branches. To create and switch to a branch, use the command

`git checkout -b new_branch_name`

To switch to an existing branch, use

`git checkout branch_name`


### Copyright

Copyright (c) 2020, Ryan DiRisio


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.

Thank you to the entire McCoy group for helping me talk through this code, with special acknowledgements to Mark Boyer and my advisor Anne McCoy
