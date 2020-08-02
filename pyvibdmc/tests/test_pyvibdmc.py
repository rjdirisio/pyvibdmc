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
    myDMC = pyvibdmc.DMC_Sim()
    assert isinstance(myDMC, pyvibdmc.DMC_Sim)

def test_runDMC():
    from ..potentials.PythonPots.harmonicOscillator1D import HODMC
    myDMC = pyvibdmc.DMC_Sim(simName="DMC_disc_test",
                 weighting='discrete',
                 initialWalkers=1000,
                 nTimeSteps=1000,
                 equilTime=50,
                 ckptSpacing=50,
                 DwSteps=50,
                 atoms=['H'],
                 dimensions=1,
                 deltaT=5,
                 potential=HODMC,
                 masses=None,
                 startStructure=Constants.convert(
                     np.array([[0.00000]]), "angstroms", to_AU=True))
    myDMC.run()
    assert True

def test_restartDMC():
    myDMC = pyvibdmc.DMC_Restart(ckptFolder='exSimResults',
                                 simName='DMC_disc_test',
                                 timeStep=500)
    myDMC.run()
    assert True