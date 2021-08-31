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
                 python_file=pyFile,
                 pot_manager=harm_pot,
                 deriv_function='first_derivative',
                 s_deriv_function='second_derivative')

    factors = np.array([1] * 49 + [0.99972538464] * 50)  # Everything except the first time step, which never changes.
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
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       DEBUG_save_training_every=1,
                       DEBUG_mass_change={'change_every': 1,
                                          'factor_per_change': factors})
    myDMC.run()
    assert True