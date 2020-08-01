"""
Unit and regression test for the potentials direcotry.
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
    omega = 3600
    x = 0.5 #some displacement in HO
    cHO = example.harmonic_oscillator(x)
    hoAns = (0.5)*Constants.mass('H')*omega**2*(x)
    assert cHO == hoAns
