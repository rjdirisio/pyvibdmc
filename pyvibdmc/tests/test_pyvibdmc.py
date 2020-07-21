"""
Unit and regression test for the pyvibdmc package.
"""
import numpy as np
from ..data import *

# Import package, test suite, and other packages as needed

import pyvibdmc
import pytest
import sys

def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules

def test_initDmcObj():
    myDMC = pyvibdmc.DMC()
    assert isinstance(myDMC, pyvibdmc.DMC)

def test_runDMC():
    def HODMC(cds):
        omega = Constants.convert(3000., 'wavenumbers', to_AU=True)
        mass = Constants.mass('H', to_AU=True)
        return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)
    myDMC = pyvibdmc.DMC(simName="DMC_disc_test",
                 weighting='discrete',
                 initialWalkers=1000,
                 nTimeSteps=1000,
                 equilTime=50,
                 chkptSpacing=250,
                 DwSteps=50,
                 atoms=['H'],
                 dimensions=1,
                 deltaT=5,
                 D=0.5,
                 potential=HODMC,
                 masses=None,
                 startStructure=Constants.convert(
                     np.array([[0.00000]]), "angstroms", to_AU=True))
    myDMC.run()
    assert True

def test_restartDMC():
    def HODMC(cds):
        omega = Constants.convert(3000., 'wavenumbers', to_AU=True)
        mass = Constants.mass('H', to_AU=True)
        return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)

    myDMC = pyvibdmc.DMC(simName="DMC_disc_test",
                         weighting='discrete',
                         initialWalkers=1000,
                         nTimeSteps=1000,
                         equilTime=50,
                         chkptSpacing=250,
                         DwSteps=50,
                         atoms=['H'],
                         dimensions=1,
                         deltaT=5,
                         D=0.5,
                         potential=HODMC,
                         masses=None,
                         startStructure=Constants.convert(
                             np.array([[0.00000]]), "angstroms", to_AU=True))
    myDMC.run()
    assert True