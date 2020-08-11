"""
Unit and regression test for the pyvibdmc package.

"""
import numpy as np
# Import package, test suite, and other packages as needed

import pyvibdmc
from ..simulation_utilities import *
import pytest
import sys
sim_ex_dir = "exSimResults"


def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules

def test_runDMC():
    import shutil,os
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    from ..potentials.PythonPots.harmonicOscillator1D import HODMC
    myDMC = pyvibdmc.DMC_Sim(sim_name="DMC_disc_test",
                             output_folder="exSimResults",
                             weighting='discrete',
                             num_walkers=1000,
                             num_timesteps=10000,
                             equil_steps=1000,
                             chkpt_every=1000,
                             wfn_every=500,
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
    myDMC = pyvibdmc.DMC_Restart(chkpt_folder=sim_ex_dir,
                                 sim_name='DMC_disc_test',
                                 time_step=1000)
    myDMC.run()
    assert True

def test_restartDMC_updateParams():
    myDMC = pyvibdmc.DMC_Restart(chkpt_folder=sim_ex_dir,
                                 sim_name='DMC_disc_test',
                                 time_step=1000)
    myDMC.num_timesteps=6000
    myDMC.initialize()

    myDMC.run()
    assert True