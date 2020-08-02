"""
Unit and regression test for the potentials directory.
"""
import numpy as np
from ..data import *
# Import package, test suite, and other packages as needed

def test_HOPot_Python():

    assert True

def test_HOPot_Fort():
    assert True

def test_HOPot_C():
    from ..potentials.CPots.exCPot import example
    omega = Constants.convert(3600,'wavenumbers',to_AU=True)
    x = 0.5 #some displacement in HO
    cHO = example.harmonic_oscillator(x)
    hoAns = (0.5)*Constants.mass('H')*omega**2*(x**2)
    assert np.around(cHO,8) == np.around(hoAns,8)