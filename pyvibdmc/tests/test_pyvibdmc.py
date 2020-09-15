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
    print(os.getcwd(),'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules

def test_runDMC():
    print(os.getcwd(),'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    import shutil
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    #initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/') # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'HODMC'
    HOpot = Potential(potential_function=potFunc,
                      python_file=pyFile,
                      potential_directory=potDir,
                      pool=2)

    myDMC = pyvibdmc.DMC_Sim(sim_name="DMC_disc_test",
                             output_folder=sim_ex_dir,
                             weighting='discrete',
                             num_walkers=1000,
                             num_timesteps=5000,
                             equil_steps=1000,
                             chkpt_every=1000,
                             wfn_every=100,
                             desc_steps=50,
                             atoms=['H'],
                             dimensions=1,
                             delta_t=5,
                             potential=HOpot,
                             masses=None,
                             start_structures=np.zeros((1,1,1))
                             )
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

# def test_restartDMC_updateParams():
#     potDir = os.path.join(os.path.dirname(__file__),
#                             '../sample_potentials/PythonPots/')
#     pyFile = 'harmonicOscillator1D.py'
#     potFunc = 'HODMC'
#     HOpot = Potential(potential_function=potFunc,
#                       python_file=pyFile,
#                       potential_directory=potDir,
#                       pool=0)
#     myDMC = pyvibdmc.DMC_Restart(potential=HOpot.getpot,
#                                  chkpt_folder=sim_ex_dir,
#                                  sim_name='DMC_disc_test',
#                                  time_step=1000)
#     myDMC.num_timesteps=6000
#     myDMC.initialize()
#
#     myDMC.run()
#     assert True