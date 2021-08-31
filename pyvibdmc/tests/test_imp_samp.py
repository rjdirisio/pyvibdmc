import pytest
import pyvibdmc as pv
import os
import numpy as np

sim_ex_dir = "imp_samp_results"

def test_run_dmc_short():
    import shutil
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                 trial_directory=potDir,
                 python_file='harm_trial_wfn.py',
                 pot_manager=harm_pot,
                 deriv_function='first_derivative',
                 s_deriv_function='second_derivative')

    myDMC = pv.DMC_Sim(sim_name="harm_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=100,
                       num_timesteps=100,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                )
    myDMC.run()
    assert True


def test_run_dmc_short_morse():

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'morse_osc_1d.py'
    potFunc = 'oh_stretch_morse'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                 trial_directory=potDir,
                 python_file='harm_trial_wfn.py',
                 pot_manager=harm_pot,
                 deriv_function='first_derivative',
                 s_deriv_function='second_derivative')

    myDMC = pv.DMC_Sim(sim_name="morse_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=1000,
                       num_timesteps=1000,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                )
    myDMC.run()
    assert True