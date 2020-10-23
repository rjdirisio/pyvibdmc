Sample Potential Energy Surface - (H2O)
==============================

This potential energy surface (PES) was developed by Harry Partridge and David W. Schwenke 
(https://doi.org/10.1063/1.473987
).

Requirements:

- gfortran (if you want to use another Fortran compiler, you will have to modify the Makefile)

This is shipped with `PyVibDMC` as a tutorial. 

In this directory, you will find:

- Fortran source code of the PES: `h2opes_v2.f`

- Fortran subroutine that calls the potential (written by Anne McCoy): `calc_h2o_pot.f`

- A Makefile that compiles the code and writes out a Python extension using `f2py`

- A Python function in `callPartridgePot.py` that calls the Python extension.

In order to use this potential, you must run `make` in this directory. If the compilation
 was successful, a `.so` file is generated. This "shared library" 
is now callable in Python, as shown in `callPartridgePot.py`. To test if it's working,
you can call `python callPartridgePot.py` and it should run without errors.