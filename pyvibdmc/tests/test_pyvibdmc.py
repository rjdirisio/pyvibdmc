import numpy as np
import sys, os
import pyvibdmc as pv
import pytest

sim_ex_dir = "exSimResults"


def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules


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
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       DEBUG_save_training_every=1,
                       DEBUG_mass_change={'change_every': 1,
                                          'factor_per_change': factors})
    myDMC.run()
    assert True


def test_run_dmc():
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

    myDMC = pv.DMC_Sim(sim_name="harm_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=1000,
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


def test_run_dmc_cont():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    myDMC = pv.DMC_Sim(sim_name="harm_osc_cont",
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
                       masses=pv.Constants.reduced_mass('O-H')
                       )
    myDMC.run()
    assert True


def test_restart_dmc():
    potDir = os.path.join(os.path.dirname(__file__),
                          '../sample_potentials/PythonPots/')
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    HOpot = pv.Potential(potential_function=potFunc,
                         python_file=pyFile,
                         potential_directory=potDir,
                         num_cores=2)
    chkpt_fold = os.path.join(os.path.dirname(__file__), '../sample_sim_data')
    myDMC = pv.dmc_restart(potential=HOpot,
                           chkpt_folder=chkpt_fold,
                           sim_name='pytest',
                           additional_timesteps=1005)
    vref = myDMC.vref_vs_tau
    walkers, cont_wts = myDMC.walkers
    myDMC.run()
    assert True


def test_mass_increase_dmc():
    potDir = os.path.join(os.path.dirname(__file__),
                          '../sample_potentials/PythonPots/')
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)
    myDMC = pv.DMC_Sim(sim_name="harm_osc_cont",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
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
                       # cont_wt_thresh=[0.002, 15],
                       masses=pv.Constants.reduced_mass('O-H') * 50,
                       DEBUG_mass_change={'change_every': 100,
                                          'factor_per_change': 0.5}
                       )
    myDMC.run()
    sim = pv.SimInfo(f'{sim_ex_dir}/harm_osc_cont_sim_info.hdf5')
    pv.Plotter.plt_vref_vs_tau(sim.get_vref())
    pv.Plotter.plt_pop_vs_tau(sim.get_pop())
    assert True
