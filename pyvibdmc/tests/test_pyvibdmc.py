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
    potFunc = 'oh_stretch_harm'
    harm_pot = Potential(potential_function=potFunc,
                         python_file=pyFile,
                         potential_directory=potDir,
                         num_cores=2)

    myDMC = pyvibdmc.DMC_Sim(sim_name="harm_osc_test",
                             output_folder=sim_ex_dir,
                             weighting='discrete',
                             num_walkers=5000,
                             num_timesteps=1000,
                             equil_steps=100,
                             chkpt_every=50,
                             wfn_every=200,
                             desc_wt_steps=199,
                             atoms=["O-H"],
                             delta_t=5,
                             potential=harm_pot,
                             log_every=50,
                             start_structures=np.zeros((1, 1, 1)),
                             cur_timestep=0)
    myDMC.run()
    assert True


def test_runDMC_cont():

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = Potential(potential_function=potFunc,
                         python_file=pyFile,
                         potential_directory=potDir,
                         num_cores=2)

    myDMC = pyvibdmc.DMC_Sim(sim_name="harm_osc_cont",
                             output_folder=sim_ex_dir,
                             weighting='continuous',
                             num_walkers=5000,
                             num_timesteps=1000,
                             equil_steps=200,
                             chkpt_every=100,
                             wfn_every=500,
                             desc_wt_steps=499,
                             atoms=["X"],
                             delta_t=5,
                             potential=harm_pot,
                             log_every=50,
                             start_structures=np.zeros((1, 1, 1)),
                             cur_timestep=0,
                             cont_wt_thresh=[0.002, 15],
                             masses = Constants.reduced_mass('O-H')
                             )
    myDMC.run()
    assert True


def test_restartDMC():
    potDir = os.path.join(os.path.dirname(__file__),
                          '../sample_potentials/PythonPots/')
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    HOpot = Potential(potential_function=potFunc,
                      python_file=pyFile,
                      potential_directory=potDir,
                      num_cores=2)

    myDMC = pyvibdmc.dmc_restart(potential=HOpot, time_step=550, chkpt_folder=sim_ex_dir, sim_name='harm_osc_test')
    myDMC.run()
    assert True
