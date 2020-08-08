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
    import os
    #TODO: remove old sim data

    from ..potentials.PythonPots.harmonicOscillator1D import HODMC
    myDMC = pyvibdmc.DMC_Sim(sim_name="DMC_disc_test",
                 weighting='discrete',
                 num_walkers=1000,
                 num_timesteps=10000,
                 equil_steps=1000,
                 chkpt_every=1000,
                 wfn_every=100,
                 desc_steps=50,
                 atoms=['H'],
                 dimensions=1,
                 delta_t=5,
                 potential=HODMC,
                 masses=None,
                 start_structures=Constants.convert(
                     np.zeros((1,1,1)), "angstroms", to_AU=True))
    myDMC.run()
    assert True

def test_restartDMC():
    myDMC = pyvibdmc.DMC_Restart(chkpt_folder='pyvibdmc/exSimResults',
                                 sim_name='DMC_disc_test',
                                 time_step=1000)
    myDMC.run()
    assert True

def test_restartDMC_updateParams():
    myDMC = pyvibdmc.DMC_Restart(chkpt_folder='pyvibdmc/exSimResults',
                                 sim_name='DMC_disc_test',
                                 time_step=1000)
    myDMC.num_timesteps=20000
    myDMC.initialize()

    myDMC.run()
    assert True