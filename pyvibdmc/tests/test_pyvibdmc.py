"""
Unit and regression test for the pyvibdmc package.

"""
import numpy as np
# Import package, test suite, and other packages as needed

import pyvibdmc
from ..simulation_utilities import *
import pytest
import sys
import os

sim_ex_dir = "exSimResults"


def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules


def test_runDMC():
    import shutil
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'HODMC'
    harm_pot = Potential(potential_function=potFunc,
                         python_file=pyFile,
                         potential_directory=potDir,
                         pool=2)

    myDMC = pyvibdmc.DMC_Sim(sim_name="DMC_Sim",
                             output_folder="harm_osc_test",
                             weighting='discrete',
                             num_walkers=5000,
                             num_timesteps=1000,
                             equil_steps=200,
                             chkpt_every=100,
                             wfn_every=100,
                             desc_wt_time_steps=100,
                             atoms=["H"],
                             delta_t=5,
                             potential=harm_pot,
                             log_every=50,
                             start_structures=np.zeros((1, 1, 1)),
                             cur_timestep=0)
    myDMC.run()
    assert True


def test_restartDMC():
    potDir = os.path.join(os.path.dirname(__file__),
                          '../sample_potentials/PythonPots/')
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'HODMC'
    HOpot = Potential(potential_function=potFunc,
                      python_file=pyFile,
                      potential_directory=potDir,
                      pool=2)

    myDMC = pyvibdmc.DMC_Restart(potential=HOpot.getpot,
                                 chkpt_folder=sim_ex_dir,
                                 sim_name='DMC_disc_test',
                                 time_step=1000)
    myDMC.run()
    assert True
